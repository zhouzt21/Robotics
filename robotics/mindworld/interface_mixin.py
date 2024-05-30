"""_summary_
Due to historical reasons, the original robot class handles both the robot and sensor/ros simulator.
However, the sensor and the ros plugins should not be built in a mind world.
"""
import numpy as np
import rclpy.time
import transforms3d
from robotics.sim.robot.controller import PID
from geometry_msgs.msg import Twist
from robotics import Pose
from std_msgs.msg import Float64MultiArray, Float64
from robotics.sim.robot.mycobot280pi import MyCobot280Arm
from robotics.ros import ROSNode


from tf2_ros.buffer import Buffer
from tf2_ros import TransformListener

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mind import Mind



class InterfaceMixin:
    mind: "Mind"
    def __init__(self, *args, node: ROSNode, **kwargs):
        super().__init__(*args, **kwargs)
        self.node = node

    def setup(self, mind):
        # setup ROS
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self.mind = mind

    def lookup_transform(self, target_frame, source_frame, time=None):
        # from target to source
        from tf2_ros import TransformException # type: ignore
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            from robotics import Pose
            from robotics.ros import transform2pose
            return transform2pose(transform)
        except TransformException:
            self.node.get_logger().warn('Cannot find transform from {} to {}'.format(target_frame, source_frame))
            return None

    def get_sensors(self):
        return {}

    def get_ros_plugins(self):
        return []

    def before_simulation_step(self):
        pass

    def after_simulation_step(self):
        pass

    
    def localization(self):
        from tf2_msgs.msg import TFMessage
        def process_tf(msg: TFMessage):
            for tf in msg.transforms:
                if tf.child_frame_id == 'base_link' and tf.header.frame_id == 'odom':
                    p = (tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z)
                    q = (tf.transform.rotation.w, tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z)
                    self._base_pose = Pose(p, q)
                    return

        self._base_pose = Pose((0, 0, 0), (1, 0, 0, 0))
        self.node.listen_once(TFMessage, 'tf', process_tf, is_static=False)
        print('initialize at pose', self._base_pose)
        self.base_tf_listener = self.node.create_subscription(TFMessage, 'tf', process_tf, 10)


class MyCobot280ArmInterface(InterfaceMixin, MyCobot280Arm):
    pid: PID
    def __init__(self, *args, node: ROSNode, move_base: bool=True, **kwargs):
        super().__init__(*args, node=node, **kwargs)
        self.move_base = move_base

    def setup(self, mind):
        super().setup(mind)

        with self.mind.acquire_state() as state:
            if state.robot_qpos is None:
                self.qpos = self.articulation.get_qpos() # type: ignore
            else:
                self.qpos = state.robot_qpos.copy()
        self.gripper_state = None

        if self.move_base:
            self.localization()

        def update_qpos(msg: Float64MultiArray):
            if self.move_base:
                self.qpos[0] = self._base_pose.p[0]
                self.qpos[1] = self._base_pose.p[1]
                self.qpos[2] = transforms3d.euler.quat2euler(np.float64(self._base_pose.q))[2] # type: ignore
                self.qpos[3:9] = np.array(msg.data)
            else:
                self.qpos[:6] = np.array(msg.data)

            if self.gripper_state is not None:
                self.qpos[-6:] = np.array(self.gripper_state)

            with self.mind._state_lock:
                self.mind._world_state.set_qpos(self.qpos.copy()) # type: ignore


        self.qpos_sub = self.node.create_subscription(Float64MultiArray, 'joint_states', update_qpos, 10)
        self.cmd_vel_pub = self.node.create_publisher(Twist, 'cmd_vel', 10)

        def send_gripper(msg: Float64MultiArray):
            self.gripper_state = np.array(msg.data)
            assert len(self.gripper_state) == 6

        self.gripper_subscriber = self.node.create_subscription(Float64MultiArray, ("/gripper_state"), send_gripper, 10)
        self.gripper_publisher = self.node.create_publisher(Float64, ("/gripper_action"), 10)

        self.qaction_pub = self.node.create_publisher(Float64MultiArray, '/joint_action', 10)

    def send_cmd_vel(self, linear: float, angular: float):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.cmd_vel_pub.publish(msg)

    def send_qaction(self, qaction: np.ndarray):
        msg = Float64MultiArray()
        msg.data = [float(x) for x in qaction] + [float(40)]
        self.qaction_pub.publish(msg)

    def set_gripper(self, val: float):
        self.gripper_publisher.publish(Float64(data=float(val)))
        

    def move_to(self, target_base_pose: np.ndarray, dist_threshold=0.02, angle_threshold=0.02, p=0.2, verbose=True):
        dx = target_base_pose[0] - self.qpos[0]
        dy = target_base_pose[1] - self.qpos[1]
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

        # if angle_to_target > np.pi / 4 or angle_to_target < -np.pi / 4:
        #     # if angle not correct, do not move
        #     if verbose:
        #         print('correct angle', angle_to_target)
        #     distance_to_target = 0
        # else:
        # move forward
        if abs(distance_to_target) < 0.01:
            # if close enough, move angle towards target
            # distance_to_target = 0.
            angle_to_target = (target_base_pose[2] - theta + np.pi) % (2 * np.pi) - np.pi
            if verbose:
                print('close enough', angle_to_target)
        else:
            if verbose:
                print('move forward', distance_to_target)
        
        if not hasattr(self, 'pid'):
            setattr(self, 'pid', PID(p, 0, 0.0))


        action = self.pid(np.array([distance_to_target, 0, angle_to_target]))
        cmd_vel = Twist()
        cmd_vel.linear.x =  action[0]
        cmd_vel.linear.y =  action[1]
        cmd_vel.angular.z = action[2]
        self.cmd_vel_pub.publish(cmd_vel)

        angle_to_target = (target_base_pose[2] - theta + np.pi) % (2 * np.pi) - np.pi
        return distance_to_target < dist_threshold and abs(angle_to_target) < angle_threshold