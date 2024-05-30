import time
import pickle
import argparse
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Float64MultiArray
from threading import Thread


class Recorder(Node):
    def __init__(self):
        super().__init__('recorder') # type: ignore
        self.qpos = []
        def update(msg: Float64MultiArray):
            self.qpos.append([time.time(), msg])
        self.qpos_sub = self.create_subscription(Float64MultiArray, 'joint_states', update, 10)
        self.base_pose = []

        def process_tf(msg: TFMessage):
            for tf in msg.transforms:
                if tf.child_frame_id == 'base_link' and tf.header.frame_id == 'odom':
                    p = (tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z)
                    q = (tf.transform.rotation.w, tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z)
                    self.base_pose.append([time.time(), (p, q)])
                    return
        self.tf_listener = self.create_subscription(TFMessage, 'tf', process_tf, 10)
        self.thread = Thread(target=rclpy.spin, args=(self,))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', type=str, default='record')
    args = parser.parse_args()

    rclpy.init()
    node = Recorder()
    try:
        node.thread.start()
        node.thread.join()
    except KeyboardInterrupt:
        with open(args.output, 'wb') as f:
            pickle.dump({'qpos': node.qpos, 'base pose': node.base_pose}, f)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()