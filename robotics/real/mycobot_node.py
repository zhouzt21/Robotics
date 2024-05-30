import rclpy
import time
from pymycobot import MyCobot
from pymycobot.genre import Angle, Coord
from pymycobot import PI_BAUD, PI_PORT
from std_msgs.msg import Float64MultiArray, Int64

import serial
import serial.tools.list_ports

# use my own driver as the default one on raspberry pi doesn't support thread lock
from pymycobot.mycobot import MyCobot



def setup():
    print("")

    plist = list(serial.tools.list_ports.comports())
    idx = 1
    for port in plist:
        print("{} : {}".format(idx, port))
        idx += 1

    _in = input("\nPlease input 1 - {} to choice:".format(idx - 1))
    port = str(plist[int(_in) - 1]).split(" - ")[0].strip()
    print(port)
    print("")

    baud = 115200
    _baud = input("Please input baud(default:115200):")
    try:
        baud = int(_baud)
    except Exception:
        pass
    print(baud)
    print("")

    DEBUG = False
    f = input("Wether DEBUG mode[Y/n]:")
    if f in ["y", "Y", "yes", "Yes"]:
        DEBUG = True
    # mc = MyCobot(port, debug=True)
    mc = MyCobot(port, baud, debug=DEBUG, thread_lock=True)
    return mc


class MyCobotNode:
    def __init__(self, name='cobot') -> None:
        self.node = rclpy.create_node(name) # type: ignore  ROS2 typing is problematic
        # self.thread = threading.Thread(target=rclpy.spin, daemon=True, args=(self.node,))

        self.publisher = self.node.create_publisher(
            Float64MultiArray, "joint_states", 10
        )
        self.ee_publisher = self.node.create_publisher(
            Float64MultiArray, "ee_states", 10
        )
        self.gripper_publisher = self.node.create_publisher(
            Int64, "gripper_states", 10
        )

        def send_radiance(msg):
            radiance = msg.data[:6]
            speed = int(msg.data[6])
            if speed > 100 or speed <= 0:
                self.node._logger.error("Speed must be between 1 and 100")
            else:
                self.mc.send_radians(radiance, speed)
                time.sleep(0.02) # wait for the command to be sent

        def send_gripper(msg: Int64):
            #self.mc.set_encoder(7, int(msg.data), 100)
            self.mc.set_gripper_value(int(msg.data), 100)
            time.sleep(0.02) # wait for the command to be sent


        self.subscriber = self.node.create_subscription(
            Float64MultiArray, "joint_action", send_radiance, 10
        )
        self.gripper_subscriber = self.node.create_subscription(
            Int64, "gripper_action", send_gripper, 10
        )

        def send_ee(msg):
            data = msg.data[:6]
            speed = int(msg.data[6])
            data = [i*1000. for i in data[:3]] + list(data[3:])
            if speed > 100 or speed <= 0:
                self.node._logger.error("Speed must be between 1 and 100")
            else:
                self.mc.send_coords(data, speed)
                time.sleep(0.02)

        self.ee_subscriber = self.node.create_subscription(
            Float64MultiArray, "ee_action", send_ee, 10
        )

        #self.mc = setup()
        self.mc = MyCobot(PI_PORT, PI_BAUD, debug=False, thread_lock=True)

        def read_radiance():
            angles = self.mc.get_angles()
            if len(angles) > 0:
                import math
                radiance = [i/180. * math.pi for i in angles]
                self.publisher.publish(Float64MultiArray(data=radiance))

            gripper = self.mc.get_encoder(7)
            self.gripper_publisher.publish(Int64(data=gripper))

            ee = self.mc.get_coords()
            self.ee_publisher.publish(Float64MultiArray(data=ee))

        self.node.create_timer(0.1, read_radiance)



if __name__ == "__main__":
    rclpy.init()
    node = MyCobotNode()
    # node.start()
    # node.join()
    while rclpy.ok():
        rclpy.spin_once(node.node)
    rclpy.shutdown()