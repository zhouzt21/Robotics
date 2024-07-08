import tqdm
from robotics.cfg import Config
import os
from sapien.utils.viewer import Viewer
import sapien.core as sapien
from typing import Optional, Union, Callable, Tuple
from robotics import Pose


from sapien import physx
from .sensors.sensor_cfg import CameraConfig
from .engines.cpu_engine import CPUEngine
from .engines.gpu_engine import GPUEngineConfig, GPUEngine

from dataclasses import dataclass
from typing import Any

import dacite
import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from .sapienpd.pd_component import PDBodyComponent, PDClothComponent
from .sapienpd.pd_config import PDConfig
from .sapienpd.pd_defs import ShapeTypes
from .sapienpd.pd_system import PDSystem

from mani_skill.envs.scene import ManiSkillScene

_Engine: CPUEngine | GPUEngine | None = None
def get_engine() -> CPUEngine | GPUEngine:
    assert _Engine is not None, "Engine is not initialized. Please create a simulator first."
    return _Engine


class SimulatorConfig(Config):
    sim_freq: int = 500
    shader_dir: str = "default"
    enable_shadow: bool = False
    contact_offset: float = 0.02

    control_freq_without_robot: int = 20

    viewer_camera: Optional[CameraConfig] = None
    viewer_id: Optional[int] = None

    gpu_config: Optional[GPUEngineConfig] = None

    solver_iterations: int = 50
    velocity_iterations: int = 1
    enable_pcm: bool = False

    n_scenes: int = 1

@dataclass
class FEMConfig:
    warp_device = "cuda:0"

    # memory config
    max_particles: int = 1 << 20
    max_constraints: int = 1 << 20
    max_constraint_total_size: int = 1 << 20
    max_colliders: int = 1 << 20

    # solver config
    sim_freq = 500
    pd_iterations = 20

    # physics config
    collision_margin = 0.2e-3
    collision_sphere_radius = 1.6e-3
    max_particle_velocity = 0.1

def gen_grid_cloth(size=(2, 1), resolution=(21, 11)):    # 是否可以作为SimalatorBase的一个方法?
    dim_x, dim_y = resolution
    xs = np.arange(dim_x)
    ys = np.arange(dim_y)
    row = np.vstack([xs[:-1], xs[1:], xs[1:] - dim_x, xs[:-1], xs[1:] - dim_x, xs[:-1] - dim_x])
    faces = (row[..., None] + ys[1:] * dim_x).reshape(6, -1).T.reshape(-1, 3)
    vertices = (np.stack(np.meshgrid(xs, ys), -1) * np.array(size) / (np.array(resolution) - 1)).reshape(-1, 2)
    vertices = np.pad(vertices, ((0, 0), (0, 1)))
    return vertices, faces


class SimulatorBase:
    """
    This is the base class to initialize the SAPIEN simulator.
    """
    _scene: sapien.Scene
    _show_camera_linesets = True
    _viewer_has_scene_updated: bool = False
    config: "SimulatorConfig"

    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.contact_offset = config.contact_offset
    ####
        fem_cfg = FEMConfig()
        interaction_links=("panda_rightfinger", "panda_leftfinger")
        cloth_size=(0.2, 0.2)
        cloth_resolution=(51, 51)
        cloth_init_pose=sapien.Pose([0.0, -0.1, 0.1])
        robot_init_qpos_noise=0

        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.interaction_links = set(interaction_links)
        self.cloth_size = cloth_size
        self.cloth_resolution = cloth_resolution
        self.cloth_init_pose = cloth_init_pose

        if isinstance(fem_cfg, FEMConfig):
            self._fem_cfg = fem_cfg
        else:
            self._fem_cfg = dacite.from_dict(data_class=FEMConfig, data=fem_cfg, config=dacite.Config(strict=True))
        #省去了BaseEnv的初始化

        assert (
            self._fem_cfg.sim_freq // self.sim_freq * self.sim_freq == self._fem_cfg.sim_freq
        ), "sim_freq must be constant multiple of sim_freq"
    ####

        self._engine = CPUEngine() if config.gpu_config is None else GPUEngine(config.gpu_config)
        global _Engine
        _Engine = self._engine

        renderer_kwargs = {}
        self._renderer = sapien.SapienRenderer(**renderer_kwargs)

        self._default_shader_dir = config.shader_dir

        shader_dir = config.shader_dir
        if shader_dir == "rt":
            sapien.render.set_camera_shader_dir("rt")
            sapien.render.set_ray_tracing_samples_per_pixel(128)
            sapien.render.set_ray_tracing_path_depth(16)
            sapien.render.set_ray_tracing_denoiser("oidn") # TODO "optix or oidn?" previous value was just True
        else:
            sapien.render.set_camera_shader_dir(shader_dir)
            sapien.render.set_viewer_shader_dir(shader_dir)

        sapien.render.set_log_level(os.getenv("MS2_RENDERER_LOG_LEVEL", "warn"))
        self._engine.set_renderer(self._renderer)

        # Set simulation and control frequency
        self._sim_freq = config.sim_freq
        self._viewer: Optional[Viewer] = None
        self.enable_shadow = config.enable_shadow
        self._configured = False

        self._viewer_camera = config.viewer_camera
        self._viewer_id = config.viewer_id
    
    
    @property
    def sim_freq(self):
        return self._sim_freq

    @property
    def sim_timestep(self):
        return 1.0 / self._sim_freq

    def set_scene(self, idx: int):
        assert 0 <= idx < self.config.n_scenes
        self._scene = self._scene_list[idx]
        self._scene_idx = idx
        self._viewer_has_scene_updated = False
        if self._viewer is not None:
            self._viewer.set_scene(self._scene)
    
    def set_viewer_scenes(self, idx: list[int], spacing: float = 1.):
        if spacing is not None and len(idx) > 1:
            import numpy as np
            side = int(np.ceil(len(idx) ** 0.5))
            idx = np.arange(len(idx))
            offsets = np.stack([idx // side, idx % side, np.zeros_like(idx)], axis=1) * spacing
        else:
            offsets = None

        self.viewer.set_scenes([self._scene_list[i] for i in idx], offsets=offsets)
        vs = self.viewer.window._internal_scene # type: ignore
        cubemap = self._scene.render_system.get_cubemap()
        if cubemap is not None:
            vs.set_cubemap(cubemap._internal_cubemap)
        else:
            vs.set_ambient_light([0.5, 0.5, 0.5])
        self._setup_viewer(False)
        self._viewer_has_scene_updated = False

    def reset(self, init_engine: bool = True):
        """
        Although this is called reset, it is actually initilize the simulator and all scenes
        """
        self.close()

        self._setup_scene()
    ####
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
            #只 BaseEnv 有 sim_cfg   如何输入 where is the input entrance of gravity?
        # self._pd_config.gravity = self.sim_cfg.scene_cfg.gravity   

        self._pd_system = PDSystem(self._pd_config, self._fem_cfg.warp_device)
        assert len(self._scene_list) == 1, "currently only single scene is supported"
        # self.scene.sub_scene    self._scene_list
        for s in self._scene_list:
            s.add_system(self._pd_system)

        self._pd_ground = PDBodyComponent(
            [ShapeTypes.GEO_PLANE],
            frictions=[1.0],
            shape2cm=[sapien.Pose(q=[0.7071068, 0, -0.7071068, 0])],
        )
        entity = sapien.Entity()
        entity.add_component(self._pd_ground)
        self._scene_list[0].add_entity(entity)  ##
    ####
        self.actor_batches = []
        self.articulation_batches = []
        self.camera_batches = []

        if self.config.n_scenes > 1:
            it = tqdm.tqdm(self._scene_list, desc="Creating all scenes")
        else:
            it = self._scene_list

        for idx, scene in enumerate(it):
            self.set_scene(idx)
        #####
        # _after_reconfigure
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
            cloth_render = sapien.render.RenderCudaMeshComponent(len(vertices), 2 * len(faces))
            cloth_render.set_vertex_count(len(vertices))
            cloth_render.set_triangle_count(2 * len(faces))
            cloth_render.set_triangles(np.concatenate([faces, faces[:, ::-1]], axis=0))
            cloth_render.set_material(sapien.render.RenderMaterial(base_color=[0.7, 0.3, 0.4, 1.0]))
            cloth_entity = sapien.Entity()
            cloth_entity.add_component(cloth_comp)
            cloth_entity.add_component(cloth_render)
            cloth_entity.set_pose(self.cloth_init_pose)

            self._scene_list[0].add_entity(cloth_entity)
            self.cloth_comp = cloth_comp
            self.cloth_render_comp = cloth_render

            self.pd_init_state = self.get_pd_state_dict()
        #####
            self._setup_lighting()
            # self._scene.load_widget_from_package("demo_arena", "DemoArena")
            self._load()

        self._scene = self._scene_list[0]

        if init_engine:
            self._engine.reset()   #gpu engine没有调用step_scenes(被注释了？)，所以step之后的after_step更新应添加在哪里？
        if len(self._scene_list) > 1:
            self._engine.init_batch(self.actor_batches, self.articulation_batches)
        
        if len(self.camera_batches) > 0:
            self._engine.set_cameras(self._scene_list, self.camera_batches)

    def _load(self):
        raise NotImplementedError

    # -------------------------------------------------------------------------- #
    # Code for setup sapien scene and renderer
    # -------------------------------------------------------------------------- #

        # ----------------don't konw if it is neccessary------------ #
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # with torch.device(self.device):
        #     self.table_scene.initialize(env_idx)

        self.set_pd_state_dict(self.pd_init_state)    

    def render_human(self):
        self.cloth_comp.update_render(self.cloth_render_comp)
        return super().render_human()

    def render(self):
        self.cloth_comp.update_render(self.cloth_render_comp)
        return super().render()

    def _load_scene(self, options: dict):
            # load sub scenes
        for s in self._scene_list:
            for e in s.entities:
                if e.name not in self.interaction_links:
                    continue
                body = e.find_component_by_type(sapien.pysapien.physx.PhysxRigidBodyComponent)
                e.add_component(PDBodyComponent.from_physx_shape(body, grid_size=3e-3))

    def _after_simulation_step(self):    ###  这是为了仿真之后更新状态 但哪里调用了仿真？
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
     # ----------------don't konw if it is neccessary------------ #

    def _get_default_scene_config(self):
        scene_config = sapien.SceneConfig()
        # note these frictions are same as unity
        physx.set_default_material(dynamic_friction=0.5, static_friction=0.5, restitution=0)
        scene_config.contact_offset = self.contact_offset
        scene_config.enable_pcm = self.config.enable_pcm
        scene_config.solver_iterations = self.config.solver_iterations
        # NOTE(fanbo): solver_velocity_iterations=0 is undefined in PhysX
        scene_config.solver_velocity_iterations = self.config.velocity_iterations
        return scene_config

    def _setup_scene(self, scene_config: Optional[sapien.SceneConfig] = None):
        """Setup the simulation scene instance.
        The function should be called in reset(). Called by `self.reconfigure`"""
        if scene_config is None:
            scene_config = self._get_default_scene_config()

        def create_scene():
            scene = self._engine.create_scene(scene_config)
            scene.set_timestep(1.0 / self._sim_freq)
            return scene

        self._scene = create_scene()

        self._scene_list = [self._scene]
        if self.config.n_scenes > 1:
            it = range(1, self.config.n_scenes)
            self._scene_list += [create_scene() for i in it]
        self._viewer_has_scene_updated = False


    def close(self):
        """Clear the simulation scene instance and other buffers.
        The function can be called in reset() before a new scene is created. 
        Called by `self.reconfigure` and when the environment is closed/deleted
        """
        self._close_viewer()
        setattr(self, "_scene", None)
        setattr(self, "_scene_list", None)

    def _close_viewer(self):
        if self._viewer is None:
            return
        self._viewer.close()
        self._viewer = None


    def _add_ground(self, altitude=0.0, render=True):
        if render:
            rend_mtl = self._renderer.create_material()
            rend_mtl.base_color = [0.06, 0.08, 0.12, 1]
            rend_mtl.metallic = 0.0
            rend_mtl.roughness = 0.9
            rend_mtl.specular = 0.8
        else:
            rend_mtl = None
        ground = self._scene.add_ground(
            altitude=altitude,
            render=render,
            render_material=rend_mtl,
        )

        self.ground = ground
        return ground


    @property
    def viewer(self):
        if self._viewer is None:
            self._viewer = Viewer(self._renderer)
            self._setup_viewer()
        return self._viewer


    def _setup_lighting(self):
        """Setup lighting in the scene. Called by `self.reconfigure`"""

        shadow = self.enable_shadow
        self._scene.set_ambient_light([0.3, 0.3, 0.3])
        # Only the first of directional lights can have shadow
        self._light1 = self._scene.add_directional_light(
            [1, 1, -1], [1, 1, 1], shadow=shadow, shadow_scale=5, shadow_map_size=2048
        )
        self._light2 = self._scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _set_viewer_camera(self, camera_config: Optional["CameraConfig"]):
        from .sensors.camera_v2 import CameraConfig
        from .sensors.sensor_base import get_pose_from_sensor_cfg

        if camera_config is None:
            camera_config = CameraConfig(p=(1, 1, 1), look_at=(0, 0, 0.5))
        import transforms3d
        pose = get_pose_from_sensor_cfg(camera_config)
        xyz, q = pose.p, pose.q
        rpy = transforms3d.euler.quat2euler(q)
        self._viewer.set_camera_xyz(*xyz) # type: ignore
        self._viewer.set_camera_rpy(rpy[0], -rpy[1], -rpy[2]) # type: ignore  mysterious

    def _setup_viewer(self, set_scene: bool=True):
        #TOOD: setup viewer
        assert self._viewer is not None
        if set_scene:
            self._viewer.set_scene(self._scene)
        camera_config = self._viewer_camera
        self._set_viewer_camera(camera_config)
        self._viewer.control_window._show_camera_linesets = self._show_camera_linesets # type: ignore

        if self._viewer_id is not None:
            assert self._viewer.plugins is not None
            for i in self._viewer.plugins:
                if hasattr(i, 'camera_index'): # HACK: set camera index, maybe there is a better way
                    i.camera_index = self._viewer_id

    def update_scene_if_needed(self):
        if not self._viewer_has_scene_updated:
            self._engine.sync_pose()
            try:
                self.viewer.window.update_render() # type: ignore
            except AttributeError:
                self._scene.update_render() # in case of old sapien version
            self._viewer_has_scene_updated = True

    def render(self, show=True):
        self.update_scene_if_needed()
        if show:
            self.viewer.render()
        return self._viewer


    # def remove_articulation(self, articulation):
    #     self._scene.remove_articulation(articulation)
    # def add_articulation(self, articulation):
    #     entities = [l.entity for l in articulation.links]
    #     for e in entities:
    #         self._scene.add_entity(e)

FrameLike = Union[str,Callable[[SimulatorBase], Tuple[str, Pose]]]


# ----------------test in main------------ #

def main():
    #scene = sapien.Scene()
    env = PickClothEnv(render_mode="human", control_mode="pd_joint_pos")    # change to create a simulator
    num_trajs = 0
    seed = 0
    env.reset(seed=seed)
    while True:
        print(f"Collecting trajectory {num_trajs + 1}, seed={seed}")
        code = solve(env, debug=False, vis=True)
        if code == "quit":
            num_trajs += 1
            break
        elif code == "continue":
            seed += 1
            num_trajs += 1
            env.reset(seed=seed)
            continue
        elif code == "restart":
            env.reset(seed=seed, options={"save_trajectory": False})


def solve(env: BaseEnv, debug=False, vis=False):
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_acc_limits=0.5,
        joint_vel_limits=0.5,
    )
    viewer = env.render_human()

    last_checkpoint_state = None
    gripper_open = True
    viewer.select_entity(sapien_utils.get_obj_by_name(env.agent.robot.links, "panda_hand")._objs[0].entity)
    for plugin in viewer.plugins:
        if isinstance(plugin, sapien.utils.viewer.viewer.TransformWindow):
            transform_window = plugin
    while True:
        transform_window.enabled = True
        # transform_window.update_ghost_objects
        # print(transform_window.ghost_objects, transform_window._gizmo_pose)
        # planner.grasp_pose_visual.set_pose(transform_window._gizmo_pose)

        env.render_human()
        execute_current_pose = False
        if viewer.window.key_press("h"):
            print("""Available commands:
            h: print this help menu
            g: toggle gripper to close/open
            n: execute command via motion planning to make the robot move to the target pose indicated by the ghost panda arm
            c: stop this episode and record the trajectory and move on to a new episode
            q: quit the script and stop collecting data and save videos
            """)
            pass
        # elif viewer.window.key_press("k"):
        #     print("Saving checkpoint")
        #     last_checkpoint_state = env.get_state_dict()
        # elif viewer.window.key_press("l"):
        #     if last_checkpoint_state is not None:
        #         print("Loading previous checkpoint")
        #         env.set_state_dict(last_checkpoint_state)
        #     else:
        #         print("Could not find previous checkpoint")
        elif viewer.window.key_press("q"):
            return "quit"
        elif viewer.window.key_press("c"):
            return "continue"
        # elif viewer.window.key_press("r"):
        #     viewer.select_entity(None)
        #     return "restart"
        # elif viewer.window.key_press("t"):
        #     # TODO (stao): change from position transform to rotation transform
        #     pass
        elif viewer.window.key_press("n"):
            execute_current_pose = True
        elif viewer.window.key_press("g"):
            if gripper_open:
                gripper_open = False
                _, reward, _, _, info = planner.close_gripper()
            else:
                gripper_open = True
                _, reward, _, _, info = planner.open_gripper()
            print(f"Reward: {reward}, Info: {info}")
        # # TODO left, right depend on orientation really.
        # elif viewer.window.key_press("down"):
        #     pose = planner.grasp_pose_visual.pose
        #     planner.grasp_pose_visual.set_pose(pose * sapien.Pose(p=[0, 0, 0.01]))
        # elif viewer.window.key_press("up"):
        #     pose = planner.grasp_pose_visual.pose
        #     planner.grasp_pose_visual.set_pose(pose * sapien.Pose(p=[0, 0, -0.01]))
        # elif viewer.window.key_press("right"):
        #     pose = planner.grasp_pose_visual.pose
        #     planner.grasp_pose_visual.set_pose(pose * sapien.Pose(p=[0, -0.01, 0]))
        # elif viewer.window.key_press("left"):
        #     pose = planner.grasp_pose_visual.pose
        #     planner.grasp_pose_visual.set_pose(pose * sapien.Pose(p=[0, +0.01, 0]))
        if execute_current_pose:
            # z-offset of end-effector gizmo to TCP position is hardcoded for the panda robot here
            result = planner.move_to_pose_with_screw(
                transform_window._gizmo_pose * sapien.Pose([0, 0, 0.102]), dry_run=True
            )
            if result != -1 and len(result["position"]) < 100:
                _, reward, _, _, info = planner.follow_path(result)
                print(f"Reward: {reward}, Info: {info}")
            else:
                if result == -1:
                    print("Plan failed")
                else:
                    print("Generated motion plan was too long. Try a closer sub-goal")
            execute_current_pose = False


if __name__ == "__main__":
    main()
