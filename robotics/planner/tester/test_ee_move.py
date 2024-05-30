from robotics.sim import Simulator, SimulatorConfig, CameraConfig, PoseConfig, MobileSmallRobot
from robotics.sim.entity import SimulatorBase
from robotics.sim.environs.ycb_clutter import YCBClutter, YCBConfig
from robotics.planner import MobileAgent


from robotics.sim.robot.mobile_small_v2 import MobileSmallV2
        
import pickle
import numpy as np
from collections import defaultdict
from robotics.sim.environ import EnvironBase 

import sapien


class BoxObstacle(EnvironBase):
    def _load(self, world: SimulatorBase):
        # return super().load()
        scene = world._scene

        # boxes
        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.02, 0.02, 0.06])
        builder.add_box_visual(half_size=[0.02, 0.02, 0.06], material=[1, 0, 0])
        self.red_cube = builder.build(name='red_cube')
        self.red_cube.set_pose(sapien.Pose([0.7, 0, 0.06]))

        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.04, 0.04, 0.005])
        builder.add_box_visual(half_size=[0.04, 0.04, 0.005], material=[0, 1, 0])
        self.green_cube = builder.build(name='green_cube')
        self.green_cube.set_pose(sapien.Pose([0.4, 0.3, 0.005]))

        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.05, 0.2, 0.1])
        builder.add_box_visual(half_size=[0.05, 0.2, 0.1], material=[0, 0, 1])
        self.blue_cube = builder.build(name='blue_cube')
        self.blue_cube.set_pose(sapien.Pose([0.55, 0, 0.1]))


    def _get_sapien_entity(self):
        return [self.red_cube, self.green_cube, self.blue_cube]



robot = MobileSmallV2(control_freq=20)
sim = Simulator(
    SimulatorConfig(viewer_camera=CameraConfig(look_at=(0., 0., 0.5), pose=PoseConfig(p=(5., 0, 5.)))), 
    robot, {'box': BoxObstacle()}
)

sim.reset()
robot.set_base_pose((-0.3, 0.), 0)

agent = MobileAgent()
agent.load(sim, robot)


from robotics.planner.skills import Sequential, move_to_pose, open_gripper, close_gripper, Compose
from robotics.planner.skills.attach_object import AttachObject


pickup_pose = [0.7, 0, 0.12, 0, 1, 0, 0]
pickup_pose[2] += 0.2

task = Sequential(
    move_to_pose(pickup_pose),
    open_gripper(),
    move_to_pose([0.7, 0., 0.2, 0., 1, 0., 0]),
    close_gripper(),
    Compose(
        AttachObject('red_cube'),
        move_to_pose([0.7, 0., 0.42, 0., 1, 0., 0], use_attach=False),
    ),
    move_to_pose([0.4, 0.3, 0.33, 0, 1, 0, 0], name="MoveTo"),
    move_to_pose([0.4, 0.3, 0.23, 0, 1, 0, 0], control_steps=1, name="Release"),
    open_gripper()
)

#task = move_to_pose(pickup_pose)


from robotics.utils import logger
from robotics.planner.skill import SkillExecutionError

with logger.configure(1, verbose=False):
    try:
        todo = [task] #, ik]
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