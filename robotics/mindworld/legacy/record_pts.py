import rclpy
import transforms3d
import numpy as np
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import PointCloud2
from robotics.semantic_map.semantic_map import msg2numpy


def main():
    rclpy.init()
    node = rclpy.create_node('test')

    msg_counter = 0
    matrix = None
    x,y,theta = None, None, None
    need_update = False
    cur_pose_t = None

    all_point_clouds = []
    all_point_colors = []

    def callback_tf(msg: TFMessage):
        nonlocal msg_counter, matrix, x, y, theta, need_update, cur_pose_t

        odom = msg.transforms[0]
        assert odom.header.frame_id == 'odom'
            
        new_x, new_y = odom.transform.translation.x, odom.transform.translation.y
        #new_theta = odom.transform.rotation.z
        quat = np.array([odom.transform.rotation.w, odom.transform.rotation.x, odom.transform.rotation.y, odom.transform.rotation.z])
        new_theta = transforms3d.euler.quat2euler(quat)[2]

        if (x is None or (np.linalg.norm([new_x - x, new_y - y]) > 0.1 or np.abs(new_theta - theta) > 0.1)):
            x, y, theta = new_x, new_y, new_theta
            need_update = True
        else:
            need_update = False

        cur_pose_t = odom.header.stamp

        matrix = np.array([
            [np.cos(new_theta), -np.sin(new_theta), 0, new_x],
            [np.sin(new_theta), np.cos(new_theta), 0, new_y],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    tf_sub = node.create_subscription(topic='/tf', msg_type=TFMessage, callback=callback_tf, qos_profile=1)

    def callback_pcd(msg: PointCloud2):
        nonlocal msg_counter, matrix, x, y, theta, need_update

        if need_update and matrix is not None:
            print(cur_pose_t, msg.header.stamp, x, y, theta)
            xyz, rgb = msg2numpy(msg, matrix)
            all_point_clouds.append(xyz)
            all_point_colors.append(rgb)

            points = np.concatenate(all_point_clouds, axis=0)
            colors = np.concatenate(all_point_colors, axis=0)

            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
            o3d.visualization.draw_geometries([pcd])

    poincloud_sub = node.create_subscription(topic='/cloud', msg_type=PointCloud2, callback=callback_pcd, qos_profile=1)

    rclpy.spin(node)


if __name__ == '__main__':
    main()