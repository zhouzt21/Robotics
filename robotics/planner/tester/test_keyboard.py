from robotics.sim import Simulator, SimulatorConfig, RobotConfig, CameraConfig, PoseConfig, MobileSmallRobot
from robotics.sim.entity import SimulatorBase
from robotics.sim.environs.ycb_clutter import YCBClutter, YCBConfig
from robotics.planner import MobileAgent, XYMoveConfig, GTXYMove, ArmIKConfig, ArmIKMove

        
import pickle
import numpy as np
from collections import defaultdict
from robotics.sim.environ import EnvironBase 

import sapien.core as sapien


class BoxObstacle(EnvironBase):
    def load(self, world: SimulatorBase):
        # return super().load()
        scene = world._scene

        # boxes
        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.02, 0.02, 0.06])
        builder.add_box_visual(half_size=[0.02, 0.02, 0.06], color=[1, 0, 0])
        self.red_cube = builder.build(name='red_cube')
        self.red_cube.set_pose(sapien.Pose([0.7, 0, 0.06]))

        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.04, 0.04, 0.005])
        builder.add_box_visual(half_size=[0.04, 0.04, 0.005], color=[0, 1, 0])
        self.green_cube = builder.build(name='green_cube')
        self.green_cube.set_pose(sapien.Pose([0.4, 0.3, 0.005]))

        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.05, 0.2, 0.1])
        builder.add_box_visual(half_size=[0.05, 0.2, 0.1], color=[0, 0, 1])
        self.blue_cube = builder.build(name='blue_cube')
        self.blue_cube.set_pose(sapien.Pose([0.55, 0, 0.1]))


    def get_actors(self):
        return [self.red_cube, self.green_cube, self.blue_cube]



robot = MobileSmallRobot(
    RobotConfig(control_freq=20, fix_root_link=True, motion_model='holonomic')
)
sim = Simulator(
    SimulatorConfig(viewer_camera=CameraConfig(look_at=(0., 0., 0.5), pose=PoseConfig(p=(5., 0, 5.)))), 
    robot, {'box': BoxObstacle()}
)

sim.reset()
robot.set_base_pose((-0.3, 0.), 0)

# import open3d as o3d
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(sim.gen_scene_pcd())
# o3d.io.write_point_cloud('test.pcd', pcd)

agent = MobileAgent()
agent.load(sim, robot)


from robotics.planner.skills import Sequential, move_to_pose, open_gripper, close_gripper, Compose, show_camera
from robotics.planner.skills.attach_object import AttachObject
from robotics.planner.skills import KeyboardController


pickup_pose = [0.7, 0, 0.12, 0, 1, 0, 0]
pickup_pose[2] += 0.2

#task = move_to_pose(pickup_pose)
task = KeyboardController()


from robotics.utils import logger
from robotics.planner.skill import SkillExecutionError

with logger.configure(1, verbose=False):
    try:
        todo = [
            task, 
            show_camera(cameras='robot_base', render_mode=['rgb', 'depth'])
        ]
        images = []
        idx = 0
        while True:
            action = agent.act({}, *todo)
            sim.step(action)
            if idx % 10 == 0:
                sim.render()
            idx += 1
            if sim._viewer is not None:
                if sim._viewer.closed:
                    break
            if task._terminated:
                break

            todo = []
    except SkillExecutionError as e:
        print(e)
        pass


messages = pickle.load(
    open("log.pkl", "rb")
)


if False:
    out = defaultdict(list)

    for k, v, t in messages:
        out[k].append((v))

    import matplotlib.pyplot as plt

    index = [0, 1, 3,4,5,6,7,8,9]

    figs, ax = plt.subplots(len(index), 1)

    out = {k: np.array(v) for k, v in out.items()}

    for i, k in enumerate(index):
        ax[i].plot(np.arange(len(out['qpos'])), out['qpos'][:, k], '.', label='qpos', color='red')
        ax[i].plot(np.arange(len(out['target'])), out['target'][:, k], '.', label='target', color='green')

    plt.legend()
    plt.show()