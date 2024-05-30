# https://mcap.dev/guides/python/ros2
import matplotlib.pyplot as plt
import transforms3d
import numpy as np

from typing import Any
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from sensor_msgs.msg import LaserScan, Image
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import PointCloud2, Image
from rosgraph_msgs.msg import Clock
from sensor_msgs_py import point_cloud2
from robotics.utils import animate

from std_msgs.msg import Float64MultiArray, MultiArrayDimension
import colorsys


def HSVToRGB(h, s, v): 
    (r, g, b) = colorsys.hsv_to_rgb(h, s, v) 
    return (int(255*r), int(255*g), int(255*b)) 
 
def getDistinctColors(n): 
    huePartition = 1.0 / (n + 1) 
    return [HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n)]

class ColorRegister:
    def __init__(self) -> None:
        self.str2color = {}
        self.tot = 0
        # Define the colormap
        cmap = plt.get_cmap('viridis')

        # Generate 20 evenly spaced values between 0 and 1
        normalized_values = np.linspace(0, 1, 20)

        # Map each normalized value to a color in the colormap
        self.distinct_colors = getDistinctColors(20)

    def __call__(self, s):
        if s not in self.str2color:
            self.str2color[s] = self.distinct_colors[self.tot % len(self.distinct_colors)]
            self.tot += 1
        return self.str2color[s]



def msg2numpy(msg: PointCloud2, matrix):
    points = point_cloud2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "r", "g", "b", "a"))
    points = np.stack([points[n] for i, n in enumerate('xyzrgba')], axis=1)
    points = np.array(points)

    xyz = points[:, :3]
    if matrix is not None:
        xyz = xyz @ matrix[:3, :3].T + matrix[:3, 3]
    rgb = points[:, 3:]
    return xyz, rgb


class SemenaticMap:
    """
    build the scene graph
    """
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    
    @classmethod
    def from_ros_bag(cls, path) -> Any:
        from robotics.semantic_map.grounded_sam import GroundSAMInterface

        groundsam = GroundSAMInterface()

        import glob
        bags = glob.glob('/root/external_home/RealRobot/realbot/ros/rosbag2_*')
        bag_file_path = sorted(bags)[-1]

        storage_options = StorageOptions(uri=bag_file_path, storage_id='sqlite3')

        converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')

        reader = SequentialReader()
        reader.open(storage_options, converter_options)
        topic_types = reader.get_all_topics_and_types()

        
        # Create a map for quicker lookup
        type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}


        msg_counter = 0
        matrix = None
        x,y,theta = None, None, None
        import tqdm
        it = tqdm.tqdm()

        tot = 0
        need_update = False

        all_point_clouds = []
        all_point_colors = []
        all_point_colors2 = []


        extrinsic = None
        intrinsic = None

        color_register = ColorRegister()

        images = []

        while reader.has_next():
            tot += 1
            # if tot > 4000:
            #     break
            it.update(1)
            topic, data, t = reader.read_next()
            msg_type = get_message(type_map[topic])
            msg_deserialized = deserialize_message(data, msg_type)
            
            if msg_type == LaserScan:
                pass
            elif msg_type == TFMessage:
                msg_deserialized: TFMessage

                odom = msg_deserialized.transforms[0]
                assert odom.header.frame_id == 'odom'
                 
                new_x = odom.transform.translation.x
                new_y = odom.transform.translation.y
                quat = np.array([odom.transform.rotation.w, odom.transform.rotation.x, odom.transform.rotation.y, odom.transform.rotation.z])
                new_theta = transforms3d.euler.quat2euler(quat)[2]

                if (x is None or (np.linalg.norm([new_x - x, new_y - y]) > 0.1 or np.abs(new_theta - theta) > 0.1)):
                    x, y, theta = new_x, new_y, new_theta
                    need_update = True
                else:
                    need_update = False

                matrix = np.array([
                    [np.cos(new_theta), -np.sin(new_theta), 0, new_x],
                    [np.sin(new_theta), np.cos(new_theta), 0, new_y],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
            elif msg_type == PointCloud2:
                msg_deserialized: PointCloud2
                pcd = msg_deserialized
                pass
                # if need_update and matrix is not None:
                #     xyz, rgb = msg2numpy(pcd, matrix)
                #     all_point_clouds.append(xyz)
                #     all_point_colors.append(rgb[..., :3])
            elif msg_type == Float64MultiArray:
                if topic == '/camera_matrix':
                    extrinsic = np.array(msg_deserialized.data).reshape((4, 4))
                else:
                    intrinsic = np.array(msg_deserialized.data).reshape((4, 4))[:3, :3]
            elif msg_type == Image:
                if topic == '/rgbd' and extrinsic is not None and intrinsic is not None and matrix is not None and need_update:
                    data = np.frombuffer(msg_deserialized.data, dtype=np.float32).reshape((msg_deserialized.height, msg_deserialized.width, 4))

                    depth = data[:, :, 3]

                    coords = np.float32(np.stack(np.meshgrid(np.arange(msg_deserialized.width), np.arange(msg_deserialized.height)), axis=-1))
                    coords = coords[:,::-1] + 0.5

                    coords = np.concatenate((coords, np.ones((msg_deserialized.height, msg_deserialized.width, 1))), axis=-1)
                    coords = coords @ np.linalg.inv(intrinsic).T

                    position = coords * depth[..., None]
                    rgb = data[:, :, :3]

                    depth_mask = depth < 0

                    position = (position @ extrinsic[:3, :3].T + extrinsic[:3, 3]) @ matrix[:3, :3].T + matrix[:3, 3]


                    rgb = (rgb * 255).astype(np.uint8)
                    output = rgb
                    all_point_clouds.append(position[depth_mask].reshape(-1, 3))
                    all_point_colors2.append(rgb[depth_mask].reshape(-1, 3)/255.)

                    image_source, boxes, logits, phrases, masks = groundsam.run(image=rgb, text_prompt='displayer,sofa,box,bottle,banana,door,cabinet,painting')
                    #print(masks)
                    output = groundsam.annotate(output, boxes, logits, phrases)
                    output = groundsam.show_mask(masks[0], output, random_color=False)
                    import cv2
                    cv2.imshow('test', output[:, :, [2, 1, 0]])
                    cv2.waitKey(1)


                    # print(rgb.shape)
                    rgb = rgb * 0 + 0.5
                    print(phrases)
                    for i in range(len(masks)-1,-1,-1):
                        #for m, a in zip(masks, phrases):
                        m = masks[i]
                        a = phrases[i]
                        if a == 'painting': continue
                        color = color_register(a)
                        print(a, color)
                        rgb[m[0]] = color[:3]

                    all_point_colors.append(rgb[depth_mask].reshape(-1, 3))

                    images.append(output)
                    

            msg_counter += 1

        print(len(all_point_clouds))

        points = np.concatenate(all_point_clouds, axis=0)
        original_color = np.concatenate(all_point_colors2, axis=0)

        if len(images) > 0:
            animate(images, 'test.mp4', fps=1)

        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if len(all_point_colors) > 0:
            print(len(all_point_colors2))
            colors = np.concatenate(all_point_colors, axis=0)
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
            #o3d.io.write_point_cloud('test.pcd', pcd)
            o3d.visualization.draw_geometries([pcd])

        pcd.colors = o3d.utility.Vector3dVector(original_color[:, :3])
        o3d.visualization.draw_geometries([pcd])

            
            
if __name__ == '__main__':
    SemenaticMap.from_ros_bag('')