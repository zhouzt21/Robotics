from __future__ import annotations
"""
Simulator built based on SAPIEN3.
"""
from typing import cast, Union, TYPE_CHECKING, Optional

import logging
import numpy as np
import sapien.core as sapien

import torch
from .entity import Entity, Composite
from .simulator_base import SimulatorBase, FrameLike, SimulatorConfig

from .cloth_env import FEMConfig, gen_grid_cloth
from .sapienpd.pd_component import PDBodyComponent, PDClothComponent
from .sapienpd.pd_config import PDConfig
from .sapienpd.pd_defs import ShapeTypes
from .sapienpd.pd_system import PDSystem

import os
import igl
import tqdm
import dacite
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.types import  SimConfig

if TYPE_CHECKING:
    from .robot.robot_base import Robot
    from .ros_plugins.module import ROSModule


class Simulator(SimulatorBase):
    _loaded_eneity: set["Entity"]
    robot: Optional["Robot"]
    elements: "Composite"

    _scenes: ManiSkillScene #MY ADD

    def __init__(
        self,
        config: SimulatorConfig,
        robot: Optional["Robot"],
        elements: dict[str, Union[Entity, dict]],
        add_ground: bool = True,
        ros_module: Optional["ROSModule"] = None,
        add_cloth: bool = False,
        fem_cfg: FEMConfig | dict = FEMConfig(),
        interaction_links=("link7", "link6","link5", "link4", 
                           "link3","link2","link1","link0"),
        cloth_size=(0.2, 0.2),
        cloth_resolution=(21, 21),
        cloth_init_pose=sapien.Pose([-2.4, 0, 1.2]),
        robot_init_qpos_noise=0,
    ):
        self.robot = robot
        self._setup_elements(elements)
        self.add_ground = add_ground and self._if_add_ground(elements)

        self.add_cloth = add_cloth  # MY ADD 
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.interaction_links = set(interaction_links)
        self.cloth_size = cloth_size
        self.cloth_resolution = cloth_resolution
        self.cloth_init_pose = cloth_init_pose

        if isinstance(fem_cfg, FEMConfig):
            self._fem_cfg = fem_cfg
        else:
            self._fem_cfg = dacite.from_dict(data_class=FEMConfig, data=fem_cfg, config=dacite.Config(strict=True))
        
        super().__init__(config)

        control_freq = self._control_freq = robot.control_freq if robot is not None else config.control_freq_without_robot
        if self._sim_freq % control_freq != 0:
            logging.warning(
                f"sim_freq({self._sim_freq}) is not divisible by control_freq({control_freq}).",
            )
        self._sim_steps_per_control = self._sim_freq // control_freq

        self.modules: list['ROSModule'] = []
        if ros_module is not None:
            self.modules.append(ros_module)
        for m in self.modules:
            m.set_sim(self)


    @property
    def control_freq(self):
        return self._control_freq

    @property
    def control_timestep(self):
        return 1.0 / self._control_freq

    @property
    def dt(self):
        return self.control_timestep

    def set_scene(self, idx: int): 
        assert 0 <= idx < self.config.n_scenes
        if self.add_cloth == True:
            self._scene = self._scenes.sub_scenes[idx]
        else:
            self._scene = self._scene_list[idx]
        self._scene_idx = idx
        self._viewer_has_scene_updated = False
        if self._viewer is not None:
            self._viewer.set_scene(self._scene)

    def reset(self, init_engine: bool = True):
        if self.add_cloth == False:
            super().reset(init_engine=False)
        else: 
            super().close()
            setattr(self, "_scenes", None)

            self._setup_scene()

            self.actor_batches = []
            self.articulation_batches = []
            self.camera_batches = []

            if self.config.n_scenes > 1:
                it = tqdm.tqdm(self._scenes.sub_scenes, desc="Creating all scenes")
            else:
                it = self._scenes.sub_scenes

            for idx, scene in enumerate(it):
                self.set_scene(idx)
                super()._setup_lighting()
                # self._scene.load_widget_from_package("demo_arena", "DemoArena")
                # super()._load()
                self._load()

            self._scene = self._scenes.sub_scenes[0]

            if init_engine:
                self._engine.reset()
            if len(self._scenes.sub_scenes) > 1:
                self._engine.init_batch(self.actor_batches, self.articulation_batches)

            if len(self.camera_batches) > 0:
                self._engine.set_cameras(self._scenes.sub_scenes, self.camera_batches)

    def _setup_scene(self, scene_config: Optional[sapien.SceneConfig] = None):
        if self.add_cloth == False:
            super()._setup_scene()
        else:    
            if scene_config is None:
                scene_config = self._get_default_scene_config()

            def create_scene():
                scene = self._engine.create_scene(scene_config)
                scene.set_timestep(1.0 / self._sim_freq)
                return scene

            self._scene = create_scene()

            self.config_man = SimConfig(sim_freq = self.config.sim_freq,
                                control_freq = self.config.control_freq_without_robot) ##MY ADD
            
            self._scenes = ManiSkillScene(sub_scenes=[self._scene], sim_cfg= self.config_man)  ##MY ADD

            if self.config.n_scenes > 1:
                it = range(1, self.config.n_scenes)
                self._scenes.sub_scenes  += [create_scene() for i in it]
            self._viewer_has_scene_updated = False

            self._pd_config = PDConfig()
            self._pd_config.max_particles = self._fem_cfg.max_particles
            self._pd_config.max_constraints = self._fem_cfg.max_constraints
            self._pd_config.max_constraint_sizes_sum = self._fem_cfg.max_constraint_total_size
            self._pd_config.max_colliders = self._fem_cfg.max_colliders
            self._pd_config.time_step = 1 / self._fem_cfg.sim_freq
            self._pd_config.n_pd_iters = self._fem_cfg.pd_iterations
            self._pd_config.collision_margin = self._fem_cfg.collision_margin
            self._pd_config.collision_sphere_radius = self._fem_cfg.collision_sphere_radius
            self._pd_config.max_velocity = self._fem_cfg.max_particle_velocity
            self._pd_config.gravity = self.config_man.scene_cfg.gravity  ##MY CHANGE   ori:sim_config

            self._pd_system = PDSystem(self._pd_config, self._fem_cfg.warp_device)
            assert len(self._scenes.sub_scenes) == 1, "currently only single scene is supported" ##scene   _scene

            self._scene = self._scenes.sub_scenes[0]  ##MY ADD

            for s in self._scenes.sub_scenes:  ##scene   _scene
                s.add_system(self._pd_system)

            self._pd_ground = PDBodyComponent(
                [ShapeTypes.GEO_PLANE],
                frictions=[1.0],
                shape2cm=[sapien.Pose(q=[0.7071068, 0, -0.7071068, 0])],
            )
            entity = sapien.Entity()
            entity.add_component(self._pd_ground)
            self._scene.add_entity(entity)  ##scene   _scene

            # cloth_path = os.path.join(os.path.dirname(__file__), "assets/deformable/trouser_sofa.obj")
            # vertices, faces = igl.read_triangle_mesh(cloth_path)
            vertices, faces = gen_grid_cloth(size=self.cloth_size, resolution=self.cloth_resolution)
        
            cloth_comp = PDClothComponent(
                vertices,
                faces,
                thickness=1e-3,
                density=1e3,
                stretch_stiffness=1e3,
                bend_stiffness=1e-3,
                friction=0.3,
                collider_iterpolation_depth=0,
            )
            near, far = 0.1, 100
            width, height = 640, 480
            camera = self._scene.add_camera('main_camera', 
                                                width=width, height=height,
                                                fovy=np.deg2rad(35), near=near, far=far,) 
            camera.set_pose(sapien.Pose([0.3, 0, 0.6], [0, 0, 0, 1]))      

            cloth_render = sapien.render.RenderCudaMeshComponent(len(vertices), 2 * len(faces))
            cloth_render.set_vertex_count(len(vertices))
            cloth_render.set_triangle_count(2 * len(faces))
            cloth_render.set_triangles(np.concatenate([faces, faces[:, ::-1]], axis=0))
            cloth_render.set_material(sapien.render.RenderMaterial(base_color=[0.7, 0.3, 0.4, 1.0]))
            cloth_entity = sapien.Entity()
            cloth_entity.add_component(cloth_comp)
            cloth_entity.add_component(cloth_render)
            cloth_entity.set_pose(self.cloth_init_pose)     

            self._scene.add_entity(cloth_entity)   ##scene   _scene
            self.cloth_comp = cloth_comp
            self.cloth_render_comp = cloth_render       
            
            self.pd_init_state = self.get_pd_state_dict()


    def _load_scene(self, options: dict):
        for s in self._scenes.sub_scenes:   ##scene   _scene
            for e in s.entities:
                if e.name not in self.interaction_links:
                    continue
                body = e.find_component_by_type(sapien.pysapien.physx.PhysxRigidBodyComponent)
                e.add_component(PDBodyComponent.from_physx_shape(body, grid_size=3e-3))

    def render_human(self):
        self.cloth_comp.update_render(self.cloth_render_comp)
        return super().render_human()

    def render(self):
        self.cloth_comp.update_render(self.cloth_render_comp)
        return super().render()

    def _after_simulation_step(self): 
        self._pd_system.sync_body()
        for _ in range(self._fem_cfg.sim_freq // self.sim_freq):
            self._pd_system.step()

    def get_pd_state_dict(self):
        return {
            "q": self._pd_system.q.numpy(),
            "qd": self._pd_system.qd.numpy(),
            "body_q": self._pd_system.body_q.numpy(),
        }

    def set_pd_state_dict(self, state):
        self._pd_system.q.assign(state["q"])
        self._pd_system.qd.assign(state["qd"])
        self._pd_system.body_q.assign(state["body_q"])

    def get_state_dict(self):
        return super().get_state_dict() | {"pd": self.get_pd_state_dict()}

    def set_state_dict(self, state):
        super().set_state_dict(state)
        self.set_pd_state_dict(state["pd"])

    def _if_add_ground(self, elements: dict[str, Union[Entity, dict]]):
        for v in elements.values():
            if hasattr(v, 'has_ground') and getattr(v, 'has_ground'):
                return False
        return True

    def _setup_elements(self, elements: dict[str, Union[Entity, dict]]):
        robot = self.robot
        if robot is not None:
            self.robot_cameras = robot.get_sensors()
            elements = dict(robot=robot, **self.robot_cameras, **elements)
        self.elements = Composite('', **elements)


    def find(self, uid=''):
        return self.elements.find(uid)

    def is_loaded(self, entity: "Entity"):
        return entity in self._loaded_eneity

    def load_entity(self, entity: "Entity"):
        if self.is_loaded(entity):
            return
        entity._load(self)
        self._loaded_eneity.add(entity)


    # load ground ..
    def _load(self):
        if self.add_ground:
            self._add_ground(render=True)
        self._loaded_eneity = set()
        self._elem_cache = {}
        self.elements.load(self)

        # TODO: maybe we can just use self._scene.get_actors() to get all the actors. However, I don't know if the order will be the same.
        self.actor_batches.append(self.elements.get_actors())
        self.articulation_batches.append(self.elements.get_articulations())
        self.camera_batches.append(self.elements.get_sapien_obj_type(sapien.render.RenderCameraComponent))

        for m in self.modules:
            m.load()


    def step(self, action: Union[None, np.ndarray, dict], print_contacts_for_debug: bool=False):
        self._viewer_has_scene_updated = False
        if action is None:  # simulation without action
            pass
        elif isinstance(action, np.ndarray) or isinstance(action, torch.Tensor):
            assert self.robot is not None
            self.robot.set_action(action)
        else:
            raise TypeError(type(action))

        #TODO: ROS_MODULE.before_control_step 
        for m in self.modules:
            m.before_control_step()

        for _ in range(self._sim_steps_per_control):
            if self.robot is not None:
                self.robot.before_simulation_step()

            if self.add_cloth == False:    
                self._engine.step_scenes(self._scene_list)
                if print_contacts_for_debug:
                    print(self._scene.get_contacts())
            else:
                self._engine.step_scenes(self._scenes.sub_scenes)  ##MY ADD
                self._after_simulation_step()
                if print_contacts_for_debug:
                    print(self._scenes.sub_scenes[0].get_contacts())  ##MY ADD

        for m in self.modules:
            m.after_control_step()
    

    # ---------------------------------------------------------------------------- #
    # Advanced: utilities for ROS2 and motion planning. I am not sure if we should 
    # put them here.
    # ---------------------------------------------------------------------------- #

    def gen_scene_pcd(self, num_points: int = int(1e5), exclude=()):
        """Generate scene point cloud for motion planning, excluding the robot"""
        pcds = []
        sim = self
        for k, v in sim.elements.items():
            if k != 'robot':
                out = v.get_pcds(num_points, exclude)
                if out is not None:
                    pcds.append(out)
        return np.concatenate(pcds)