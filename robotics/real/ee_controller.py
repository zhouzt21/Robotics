"""_summary_
Node for teleop the elephant robo
Planning Node ..
"""
import argparse
import time
import transforms3d
import numpy as np
import os

from typing import Optional
from robotics import Pose

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import Twist

from robotics.real.ik import IKSolver


import rclpy
from rclpy.node import Node
from robotics.sim import Simulator, SimulatorConfig
from robotics.sim.robot.mycobot280pi import MyCobot280Arm

from std_msgs.msg import Float64MultiArray, Int64



class TeleopNode(Node):
    def __init__(self, sim: Simulator, 
                 arm: MyCobot280Arm, 
                 ik: IKSolver, 
                 move_base: bool = False,
                 plan_base_ik: bool = False,
    ):

        super().__init__('teleop_node') # type: ignore
        self.sim: Simulator = sim
        self.arm = arm
        self.ik = ik
        self.plan_base_ik = plan_base_ik

        self.qpos = self.arm.articulation.get_qpos()
        self.gripper_state = None
        def update_qpos(msg: Float64MultiArray):
            if self.move_base:
                self.qpos[0] = self.base_pose.p[0]
                self.qpos[1] = self.base_pose.p[1]
                self.qpos[2] = transforms3d.euler.quat2euler(np.float64(self.base_pose.q))[2] # type: ignore
                self.qpos[3:9] = np.array(msg.data)
            else:
                self.qpos[:6] = np.array(msg.data)
            self.arm.articulation.set_qpos(self.qpos)
            ee_pose = self.ik.ee_link.pose
            self.ee_pose_pub.publish(Float64MultiArray(data=list(ee_pose.p) + list(ee_pose.q) + list(self.base_pose.q)))

        def update_gripper(msg: Int64):
            self.gripper_state = msg.data

        self.qpos_sub = self.create_subscription(Float64MultiArray, 'joint_states', update_qpos, 10)
        self.joint_action_pub = self.create_publisher(Float64MultiArray, 'joint_action', 10)

        self.gripper_sub = self.create_subscription(Int64, 'gripper_states', update_gripper, 10)
        self.gripper_pub = self.create_publisher(Int64, 'gripper_action', 10)

        self.ee_pose_pub = self.create_publisher(Float64MultiArray, 'ee_pose', 10)

        self.move_base = move_base

        if self.move_base:
            #self.buffer = Buffer()
            #self.lookuper = TransformListener(self.buffer, self)
            from tf2_msgs.msg import TFMessage

            def process_tf(msg: TFMessage):
                for tf in msg.transforms:
                    if tf.child_frame_id == 'base_link' and tf.header.frame_id == 'odom':
                        p = (tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z)
                        q = (tf.transform.rotation.w, tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z)
                        self.base_pose = Pose(p, q)
                        return
            self.tf_listener = self.create_subscription(TFMessage, 'tf', process_tf, 10)


            self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
            self.base_pose = Pose((0, 0, 0), (1, 0, 0, 0))

            
        self.ee_target_sub = self.create_subscription(Float64MultiArray, 'ee_target', self.ee_target_callback, 10)
        self.target_base_pose = None
        self.speed = 60

        # self.last_ee_target = None
        # self.last_qpos_target = None

        
    def ee_target_callback(self, msg: Float64MultiArray):
        ee_target = Pose(msg.data[:3], msg.data[3:])
        qpos = self.qpos
        # print('target_callback', self.base_pose, self.qpos, ee_target)
        target_qpos, arm_action = self.ik.step(
            ee_target, qpos, self.move_base and self.plan_base_ik, 
            base_pose=self.base_pose
        ) #, self.last_ee_target, self.last_qpos_target)

        if target_qpos is not None:
            if self.move_base and self.base_pose is not None and self.plan_base_ik:
                self.target_base_pose = target_qpos[:3]
                target_qpos = target_qpos[3:]

            action = [float(i) for i in arm_action] + [self.speed]
            self.joint_action_pub.publish(Float64MultiArray(data=action))


    def start(self, hz=30, speed=30):
        self.speed = speed
        self.ik.setup_sapien()

        from threading import Thread
        self.thread = Thread(target=rclpy.spin, args=(self,))
        self.thread.start()

        dt = 1 / hz

        from robotics.sim.robot.controller import PID

        self.pid = PID(0.2, 0.0, 0.0)
        while True:
            cur = time.time()
            if self.move_base:
                if self.target_base_pose is not None:
                    dx = self.target_base_pose[0] - self.qpos[0]
                    dy = self.target_base_pose[1] - self.qpos[1]
                    theta = self.qpos[2]

                    distance_to_target = np.sqrt(dx**2 + dy**2)
                    angle_to_target = np.arctan2(dy, dx) - theta
                    angle_to_target = (angle_to_target + np.pi) % (2 * np.pi) - np.pi  # Normalize angle

                    if angle_to_target < -np.pi / 2 or angle_to_target > np.pi / 2:
                        if angle_to_target < -np.pi / 2:
                            angle_to_target = angle_to_target + np.pi
                        else:
                            angle_to_target = angle_to_target - np.pi
                        distance_to_target = -distance_to_target

                    if abs(distance_to_target) < 0.02:
                        distance_to_target = 0
                        angle_to_target = (self.target_base_pose[2] - theta + np.pi) % (2 * np.pi) - np.pi
                    elif angle_to_target > np.pi / 4 or angle_to_target < -np.pi / 4:
                        distance_to_target = 0


                    action = self.pid(np.array([distance_to_target, 0, angle_to_target]))
                    cmd_vel = Twist()
                    cmd_vel.linear.x =  action[0]
                    cmd_vel.linear.y =  action[1]
                    cmd_vel.angular.z = action[2]
                    self.cmd_vel_pub.publish(cmd_vel)
                


            self.sim.viewer.render()

            time.sleep(max(dt - (time.time() - cur), 0.))
            if self.sim.viewer.closed:
                break


def run_teleop(domain_id):
    import subprocess
    import sys
    process = subprocess.Popen(['python3', '-m', 'realbot.real.teleop_vive', str(domain_id)], stdout=sys.stdout.fileno(), stderr=sys.stdout.fileno())
    return process


        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fix_base', action='store_true')
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--unbounded', action='store_true')
    args = parser.parse_args()


    domain_id = str(1 if not args.real else 0)
    if not args.real:
        os.environ["ROS_DOMAIN_ID"] = domain_id
    else:
        os.system("ros2 param set /motion_control safety_override full")

    job = run_teleop(domain_id)
    try:

        from robotics.sim.simulator import CameraConfig, CameraV2

        arm = MyCobot280Arm(60, move_base=not args.fix_base)

        camera = CameraV2(CameraConfig(look_at=(0., 0., 0.2), p=(-0.6, 0.0, 1.3), base='robot/base_link'))

        sim = Simulator(
            SimulatorConfig(viewer_camera=CameraConfig(p=(-0.3, 0., 1.), q=(0.5, -0.5, 0.5, -0.5))),
            arm, 
            {'camera': camera}
        )
        sim.reset()
        sim.viewer.plugins[2].camera_index=1 # type: ignore
        ik_solver = IKSolver(sim, 'gripper_base',
                    np.array([[0.2, -0.2, 0.12],[0.4, 0.2, 0.32]]) if not args.unbounded else None)

        rclpy.init(args=None)
        node = TeleopNode(sim, arm, ik_solver, move_base=not args.fix_base)
        node.start(speed=60, hz=30)

        node.destroy_node()
        rclpy.shutdown()

    finally:
        job.kill()
        
        
if __name__ == '__main__':
    main()