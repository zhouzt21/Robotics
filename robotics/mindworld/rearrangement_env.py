import os
import argparse

import time
from typing import Dict, Optional
import numpy as np
from robotics.sim import Simulator, SimulatorConfig, CameraConfig
from robotics.sim.robot.mycobot280pi import MyCobot280Arm
from robotics.sim.environs.ycb_clutter import YCBClutter, YCBConfig
from robotics.sim.environs.sapien_square_wall import SquaWall, WallConfig
from robotics.sim.environs.thor import ThorEnv, ThorEnvConfig

from robotics.ros import ROSNode
import argparse


from robotics.sim.ros_plugins import ROSPlugin
from robotics.sim.simulator import Simulator

from std_msgs.msg import Float64MultiArray, MultiArrayDimension


class ObjectPosePublisher(ROSPlugin):
    def __init__(self, ycb: YCBClutter, frame_mapping: Dict[str, str] | None = None) -> None:
        super().__init__(frame_mapping)
        self.ycb = ycb

    def _load(self, world: Simulator, node: ROSNode):
        self.publisher = node.create_publisher(Float64MultiArray, 'gt_object_pose', 10)

    
    def after_step(self, world: Simulator):
        #return super().after_step(world)
        n = len(self.ycb._actors)
        msg = Float64MultiArray()
        data = []
        for i in range(n):
            actor = self.ycb._actors[i]
            model_id, scale = self.ycb._model_id[i]
            pose = actor.get_pose()
            data.append(
                [
                    model_id, scale, pose.p[0], pose.p[1], pose.p[2], pose.q[0], pose.q[1], pose.q[2], pose.q[3]
                ]
            )
        msg.data = np.array(data).astype(np.float32).flatten().tolist()
        self.publisher.publish(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--not_view', action='store_true')
    parser.add_argument('--ros_domain_id', type=int, default=1)
    parser.add_argument('--scene', type=str, default='wall', choices=['wall', 'thor'])
    args = parser.parse_args()
    os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain_id)

    node = ROSNode('cobot_sim', use_sim_time=True)
    ycb = YCBClutter(YCBConfig(bbox=(-2., -2., 2., 2.)), keyword='-a_')
    if args.scene == 'thor':
        scene = ThorEnv(ThorEnvConfig())
    else:
        scene = SquaWall(WallConfig())

    environments = {"scene": scene, 'ycb': ycb}

    sim = Simulator(
        SimulatorConfig(viewer_camera=CameraConfig(look_at=(0., 0., 0.5), p=(0.5, 0.5, 1.))),
        MyCobot280Arm(60, arm_controller='posvel'), environments, ros_node=node
    )
    sim._show_camera_linesets=False
    sim.register_ros_plugins(ObjectPosePublisher(ycb))

    sim.reset()
    dt = sim.dt

    idx = 0
    while (args.not_view) or (not sim.viewer.closed):
        cur = time.time()
        idx += 1
        sim.step(None)
        if not args.not_view:
            sim.render()
        time.sleep(max(dt - (time.time() - cur), 0.))

        
if __name__ == '__main__':
    main()