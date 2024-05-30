import rclpy
import pickle
import argparse
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import time
from robotics.sim.robot.mycobot280pi import MyCobot280Arm
from robotics.sim import Simulator, SimulatorConfig
from robotics.sim.simulator import CameraConfig, CameraV2
from threading import Thread



class Player(Node):
    def __init__(self):
        super().__init__('publisher') # type: ignore
        self.qpos = []
        self.qpos_pub = self.create_publisher(Float64MultiArray, 'joint_action', 10)
        self.base_pose = []
        self.thread = Thread(target=rclpy.spin, args=(self,))



def main():
    # arm = MyCobot280Arm(60, move_base=False)

    # camera = CameraV2(CameraConfig(look_at=(0., 0., 0.2), p=(-0.6, 0.0, 1.3), base='robot/base_link'))

    # sim = Simulator(
    #     SimulatorConfig(viewer_camera=CameraConfig(p=(-0.3, 0., 1.), q=(0.5, -0.5, 0.5, -0.5))),
    #     arm, 
    #     {'camera': camera}
    # )
    # sim.reset()
    rclpy.init()

    node = Player()
    node.thread.start()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default='record')
    parser.add_argument('--start', '-s', type=float, default=0.)
    args = parser.parse_args()


    with open(args.input, 'rb') as f:
        data = pickle.load(f)

    qpos = data['qpos']
    init = qpos[0][0]
    start_time = time.time()
    for t, q in qpos:
        #if init is None or t - init > 0.01:
        #    pass
        if t < init + args.start:
            continue
        passed = time.time() - start_time
        if init is not None and (t - init -args.start) > passed:
            time.sleep((t - init - args.start) - passed)
        print(t - init)

        # print(t, q)
        # if init is not None:
        #     print(t - init, time.time() - start_time)
        node.qpos_pub.publish(Float64MultiArray(data=[*list(q.data), 40]))
    node.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()