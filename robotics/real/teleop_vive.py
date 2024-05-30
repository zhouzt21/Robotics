import transforms3d
import numpy as np
from typing import Optional
from robotics import Pose

from geometry_msgs.msg import Twist

import rclpy
import numpy as np
from rclpy.node import Node
from robotics.teleop import triad_openvr
from std_msgs.msg import Float64MultiArray, Int64


class ViveController:

    def __init__(self) -> None:
        self.triad = triad_openvr.triad_openvr()
        self.controller = self.triad.devices["controller_1"]
        self.p0 = None
        self.last_p_abs = None
        self.engaged = False

    def reset(self):
        self.p0 = None
        self.last_p_abs = None
        self.engaged = False

    def get_controller_inputs(self):
        return self.controller.get_controller_inputs()
    
    def get_button_state(self):
        inputs_dict = self.controller.get_controller_inputs()
        # trackpad_touched = \
        #     not np.allclose(np.array([inputs_dict["trackpad_x"], inputs_dict["trackpad_y"]]), np.array([0, 0]))
        #print(inputs_dict)
        return inputs_dict
        # # trackpad_touched = True
        # return dict(
        #     trigger=inputs_dict.pop("trigger"], 
        #     trackpad_pressed=inputs_dict["trackpad_pressed"],
        #     trackpad_x=inputs_dict["trackpad_x"],
        #     trackpad_y=inputs_dict["trackpad_y"],
        #     trackpad_touched=trackpad_touched,
        # )
    
    def get_pose_matrix(self):
        """ preprocess HTC vive pose to match user's reference frame """
        pose_mat = self.controller.get_pose_matrix()
        if pose_mat is None:
            return None, None
        pose_mat = np.array([[pose_mat[i][j] for j in range(4)] for i in range(3)])
        pose_mat = np.array(
            [
                [ 0,  0, -1], 
                [-1,  0,  0], 
                [ 0,  1,  0]
            ]
        ) @ pose_mat
        pose_r, pose_t = pose_mat[:3, :3], pose_mat[:3, 3]
        return pose_r, pose_t
    
    def get_pose(self) -> Optional[Pose]:
        """ raw vive pose """
        pose_r, pose_t_abs = self.get_pose_matrix()
        if pose_r is None or pose_t_abs is None:
            return None
        pose_quat = transforms3d.quaternions.mat2quat(pose_r)
        pose_t = pose_t_abs
        return Pose(pose_t, pose_quat)
    
    def get_pose_ee(self) -> Optional[Pose]:
        """ matching robot ee rotation """
        pose_raw = self.get_pose()
        if pose_raw is None:
            return None
        rot = transforms3d.quaternions.quat2mat(pose_raw.q)
        rot[:, 0], rot[:, 1], rot[:, 2] = -rot[:, 1], -rot[:, 0], -rot[:, 2]
        quat = transforms3d.quaternions.mat2quat(rot)
        pose_ee = Pose(pose_raw.p, quat)
        return pose_ee

    def get_gripper_qpos(self) -> float:
        button_states = self.get_button_state()
        trigger = button_states["trigger"]
        gripper_hi, gripper_lo = 850, 0
        gripper_pos = (1 - trigger) * (gripper_hi - gripper_lo) + gripper_lo
        return gripper_pos



class TeleopNode(Node):
    def __init__(self):
        super().__init__('vive_node') # type: ignore
        self.vive_controller = ViveController()

        self.gripper_pub = self.create_publisher(
            Int64, 'gripper_action', 10
        )
        self.ee_pose = None
        self.base_pose_q = None
        def update_ee(msg: Float64MultiArray):
            self.ee_pose = Pose(msg.data[:3], msg.data[3:7])
            self.base_pose_q = msg.data[7:]
        self.ee_sub = self.create_subscription(
            Float64MultiArray, 'ee_pose', update_ee, 10
        )

        self.ee_target_pub = self.create_publisher(
            Float64MultiArray, 'ee_target', 10
        )


        self.ee_states = None
        def read_ee(msg: Float64MultiArray):
            self.ee_states = msg.data

        self.ee_states_sub = self.create_subscription(
            Float64MultiArray, 'ee_states', read_ee, 10
        )

        self.ee_action_pub = self.create_publisher(
            Float64MultiArray, 'ee_action', 10
        )

        self.joint_action_pub = self.create_publisher(
            Float64MultiArray, 'joint_action', 10
        )


        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)



    def start(self, hz=30, speed=30, scale=1.):
        from threading import Thread
        self.thread = Thread(target=rclpy.spin, args=(self,))
        self.thread.start()
        dt = 1 / hz

        import time

        rot = Pose((0, 0, 0), transforms3d.euler.euler2quat(0, 0, np.pi))
        T = np.array(
            [[0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1]]
        )
        rot2 = Pose(T).inv()
        def transform(pose):
            """
            code for transfrom the coordinate system
            I don't know why it works
            """
            return rot * pose * rot2

        last_gripper = None
        grasped = False

        init = None
        print("press trackpad to start")
        last_press = None

        arm_mode = False
        TOT = 0

        while True:
            cur = time.time()
            button_states = self.vive_controller.get_button_state()
            # self.sim.viewer.render()
            if button_states['ulButtonPressed'] == 4:

                joint_array = [
                    # 2.07,
                    # -0.07,
                    # -1.28,
                    # -0.29,
                    # -0.45,
                    # 0.978,

                    -1.26,
                    -1.67 + 0.1,
                    -0.63,
                    0.75,
                    -1.47,
                    -0.62,
                    40
                ]

                #pose1 = [
                #    -1.302190154912969, -0.7438593271999832, -1.1550588989698474, 0.3680899392456041, -0.061261056745000965, -0.3389429407372988, 30.0]
                pose1 = joint_array


                self.joint_action_pub.publish(Float64MultiArray(data=pose1))

            if button_states['trackpad_pressed'] and (last_press is None or time.time() - last_press > 0.5):

                joint_array = [
                    # -1.2807226051134388,
                    # -1.3590878885279845,
                    # -1.293114442802599,
                    # 1.8191566793536895,
                    # -0.23300145514124296,
                    # 0.9248499706317953,
                    # -1.32,
                    # -0.65,
                    # -0.737,
                    # -0.16,
                    # -0.21,
                    # -0.72,
                    -1.29,
                    -1.38,
                    0.,
                    -0.22,
                    #3.16,
                    -1.4,
                    -0.55,

                    # -0.88,
                    # -1.16,
                    # -1.42,
                    # 1.57,
                    # -0.64,
                    # -0.52,
                    40,
                ]
                self.joint_action_pub.publish(Float64MultiArray(data=joint_array))

            if button_states['menu_button'] and (last_press is None or time.time() - last_press > 0.5):
                arm_mode = not arm_mode
                if arm_mode:
                    print("switch to arm mode")
                else:
                    print("switch to base mode")
                last_press = time.time()

            if self.ee_pose is not None:
                pose = self.vive_controller.get_pose_ee()

                def send_target(pose, minus=False):
                    nonlocal init
                    init_vive, ee, TT = init
                    delta_pose = transform(init_vive).inv() * transform(pose)
                    delta_pose.p = delta_pose.p
                    ee_target = TT * delta_pose * TT.inv() * ee

                    out = list(ee_target.p) + list(ee_target.q)
                    if minus:
                        #out[2] = max(out[2] - 0.1, 0.11)
                        out[2] = 0.10
                        print(out)
                    self.ee_target_pub.publish(Float64MultiArray(data=out))

                TOT += 1
                if TOT % 10 == 0:
                    if grasped:
                        self.gripper_pub.publish(Int64(data=1))
                    else:
                        self.gripper_pub.publish(Int64(data=255))


                # gripper
                if last_gripper is None or time.time() - last_gripper > 0.3:
                    if button_states['trigger'] > 0.5:
                        if not grasped:
                            grasped = True

                            # if pose is not None and init is not None:
                            #     print('sent...')
                            #     send_target(pose, minus=True)
                            #     time.sleep(4.)

                            self.gripper_pub.publish(Int64(data=1))
                            self.gripper_pub.publish(Int64(data=1))
                            self.gripper_pub.publish(Int64(data=1))
                            print('close gripper')

                            # if pose is not None and init is not None:
                            #     time.sleep(2.)
                            #     send_target(pose)
                            #     time.sleep(4.)
                            #     arm_mode = False
                            #     print('back to base mode')

                        elif grasped:
                            grasped = False
                            self.gripper_pub.publish(Int64(data=255))
                            self.gripper_pub.publish(Int64(data=255))
                            self.gripper_pub.publish(Int64(data=255))
                            print('open gripper')
                    last_gripper = time.time()

                # pose
                if pose is not None:
                    if not arm_mode:
                        init = None

                    if arm_mode:
                        if init is None:
                            init = (pose, self.ee_pose, Pose(self.ee_pose.p, self.base_pose_q))
                            # import copy
                            # self.delta = Pose()

                        if init is not None:
                            send_target(pose)

            x = button_states["trackpad_x"]
            y = button_states["trackpad_y"]

            if not arm_mode:
                x, y = y, x
                y = -y

                distance_to_target = np.sqrt(x**2 + y**2)
                angle_to_target = np.arctan2(y, x)
                # print(x, y, distance_to_target, angle_to_target)
                angle_to_target = (angle_to_target + np.pi) % (2 * np.pi) - np.pi  # Normalize angle

                if angle_to_target < -np.pi / 2 or angle_to_target > np.pi / 2:
                    if angle_to_target < -np.pi / 2:
                        angle_to_target = angle_to_target + np.pi
                    else:
                        angle_to_target = angle_to_target - np.pi
                    distance_to_target = -distance_to_target

                cmd_vel = Twist()
                cmd_vel.linear.x =  distance_to_target * np.cos(angle_to_target) * 0.35
                cmd_vel.linear.y =  0.
                cmd_vel.angular.z = angle_to_target * 0.35
                self.cmd_vel_pub.publish(cmd_vel)
            # else:
            #     if self.ee_states is not None and np.linalg.norm([x, y]) > 0.01:
            #         ee_target = self.ee_states[:6]
            #         if len(ee_target) > 0:
            #             ee_target[0] += x * 10
            #             ee_target[1] += y * 10
            #             ee_target = [i/1000. for i in ee_target[:3]] + list(ee_target[3:])
            #             self.ee_action_pub.publish(Float64MultiArray(data=[i for i in ee_target ]+ [40]))



            if time.time() - cur < dt:
                time.sleep(dt - (time.time() - cur))

        
def main():
    import sys
    import os
    if len(sys.argv) > 1:
        os.environ["ROS_DOMAIN_ID"] = sys.argv[1]
        print("ROS DOMAIN ID", os.environ["ROS_DOMAIN_ID"])
    rclpy.init(args=None)
    node = TeleopNode()
    node.start(speed=60, hz=30, scale=1.)
    node.destroy_node()
    rclpy.shutdown()
        
        
if __name__ == '__main__':
    main()