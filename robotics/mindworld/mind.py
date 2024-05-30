import numpy as np
import time
import threading
from robotics import Pose
from robotics.ros import ROSNode
from std_msgs.msg import String

from .worldstate import WorldState

import sapien

from .interface_mixin import MyCobot280ArmInterface
from robotics.sim.simulator import Simulator, SimulatorConfig
from robotics.utils.sapien_utils import get_rigid_dynamic_component
from contextlib import contextmanager

from typing import Optional, Union, TYPE_CHECKING, List
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .plugin import Plugin



FACTORY = {
    'mycobot280pi': lambda node : MyCobot280ArmInterface(60, node=node, arm_controller='posvel'),
}


class Mind:
    """
    Build a mind for robot, which is essentially a simulator or a digital twin.

    Operator over the Mind in other thread must be protected by the lock.

    NOTE: we can also call it Lingjing or fairyland.
    """
    robot_name: str

    plugins: List['Plugin']
    hz: Optional[float] = None

    def __init__(self, ROS_DOMAIN_ID: Optional[int]=None, plugins: Optional[List['Plugin']]=None, use_sim_time: bool=False):
        self.node = ROSNode('mindworld', use_sim_time=use_sim_time, ros_domain_id=ROS_DOMAIN_ID)

        def set_robot_name(msg: String):
            self.robot_name = msg.data
        self.node.listen_once(String, '/robot_name_static', set_robot_name, is_static=True)
        
        self._state_lock = threading.Lock()
        self._world_state_ = WorldState(self.robot_name, None)
        self._actors: List[sapien.Entity] = []


        self.robot = FACTORY[self.robot_name](self.node)
        sim_config = SimulatorConfig(shader_dir='point')

        self.sim = Simulator(sim_config, self.robot, {}, add_ground=True)
        self.sim.reset()
        self.scene = self.sim._scene

        self.robot.setup(self)


        self.plugins = plugins or []
        for plugin in self.plugins:
            plugin.setup(self)


    @contextmanager
    def acquire_state(self):
        with self._state_lock:
            yield self._world_state_

    @property
    def viewer(self):
        return self.sim.viewer

    def render(self, hz: Optional[float]=None, render: bool=True, step: bool=True):
        hz = hz or self.hz

        if step:
            self.step()
        for i in self.plugins:
            i.before_render_step()

        if hz is not None:
            t = time.time()
            if hasattr(self, '_last_render'):
                time.sleep(max(0, 1 / hz - (t - getattr(self, '_last_render'))))
            setattr(self, '_last_render', t)
        return self.sim.render(show=render)

        
    def synchronize(self, world_state: WorldState):
        if world_state.robot_qpos is not None:
            self.robot.articulation.set_qpos(world_state.robot_qpos) # type: ignore
        if len(self._actors) < len(world_state.object_list):
            for i in range(len(self._actors), len(world_state.object_list)):
                cad = world_state.object_list[i]
                actor = cad.load_into_simulator(self.sim)
                self._actors.append(actor)

        for i in range(len(world_state.object_list)):
            self._actors[i].set_pose(world_state.object_pose[i])
        world_state.updated = False


    def step(self, action: Optional[NDArray]=None):
        # NOTE: action is not used for now
        # required for synchronization

        if action is not None:
            return self.sim.step(action)

        with self.acquire_state() as state:
            if state.updated:
                self.synchronize(state)
    
    def add_pointcloud(self, points, colors: Optional[NDArray]=None, scales: Union[NDArray, float]=0.002, capacity: int=0):
        # https://github.com/haosulab/SAPIEN/blob/139062d33738a382a335c532652860d35d42981f/python/pybind/sapien_renderer.cpp#L736-L740
        import sapien.render

        # capacity is the maximum number of points. 0 means set by the first number of points
        pcd = sapien.render.RenderPointCloudComponent(capacity)
        pcd.set_vertices(points)

        if colors is not None:
            if colors.shape[1] == 3:
                colors = np.concatenate([colors, np.ones((colors.shape[0], 1))], axis=1)
            assert colors.shape[1] == 4
            pcd.set_attribute("color", colors) # type: ignore

        pcd.set_attribute("scale", np.broadcast_to(scales, (points.shape[0],)))  # type: ignore
        actor = self.scene.create_actor_builder().build()
        actor.add_component(pcd)
        actor.set_pose(Pose([0, 0, 0], [1, 0, 0, 0]))
        return actor, pcd

        
    def get_robot_frame(self) -> Pose:
        if hasattr(self.robot, '_base_pose'):
            return getattr(self.robot, '_base_pose')
        raise NotImplementedError

    def gen_pcd(self, n=1000):
        from robotics.utils.sapien_utils import get_actor_mesh
        points = []
        for actor in self._actors:
            if get_rigid_dynamic_component(actor) is None:
                continue
            mesh = get_actor_mesh(actor)
            if mesh is None:
                continue
            points.append(mesh.sample(n))
        return np.concatenate(points, axis=0)

        
