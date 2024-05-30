import rclpy
import threading
import argparse
import cv2
import numpy as np
import numpy
from rclpy.node import Node
from typing import Dict
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from robotics.utils.camera_utils import CameraInfoPublisher


class CameraAPI:
    def setup(self) -> Dict:
        raise NotImplementedError

    def capture(self):
        pass



import pyrealsense2 as rs
class Realsense(CameraAPI): 
    def setup(self) -> Dict:
        pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        H = 480
        W = 848

        config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
        config.enable_stream(rs.stream.infrared, 1)  # Infrared stream 1 (Left IR)
        config.enable_stream(rs.stream.infrared, 2)  # Infrared stream 2 (Right IR)
        config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

        # Start streaming
        profile = pipeline.start(config)
        profile.get_device().sensors[1].set_option(rs.option.exposure, 86)
        align_to = rs.stream.color
        align = rs.align(align_to)
        for _ in range(10):  # wait for white balance to stabilize
            frames = pipeline.wait_for_frames()

        self.pipeline = pipeline
        self.profile = profile
        self.align = align

        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        K = np.array([[intr.fx, 0, intr.ppx],
                      [0, intr.fy, intr.ppy],
                      [0, 0, 1]])
        return {
            'intrinsic': K,
        }


    def capture(self, return_ir=False):
        pipeline, profile, align = self.pipeline, self.profile, self.align
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            depth = np.asanyarray(frames.get_depth_frame().get_data())

            output = {'rgb': rgb, 'depth': depth}

            if return_ir:
                ir_frame1 = frames.get_infrared_frame(1)  # Left IR frame
                ir_frame2 = frames.get_infrared_frame(2)  # Right IR frame
                if ir_frame1 and ir_frame2:
                    output['ir_l'] = np.asanyarray(ir_frame1.get_data())
                    output['ir_r'] = np.asanyarray(ir_frame2.get_data())
            yield output
            
            
class OpenCV(CameraAPI):
    def setup(self) -> Dict:
        cap = cv2.VideoCapture(4)
        self.cap = cap
        self.meta = {
            'cap': cap
        }
        return self.meta

    def capture(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield {'rgb': rgb}


class CameraNode(Node):

    def __init__(self, api: CameraAPI, name: str='camera_node'):
        super().__init__(node_name=name)
        from sensor_msgs.msg import Image
        self.publisher = self.create_publisher(Image, '/rgb', 10)
        self.depth_publisher = self.create_publisher(Image, '/depth', 10)
        self.api = api
        self.meta = self.api.setup()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action='store_true')
    args = parser.parse_args()

    rclpy.init()
    api = Realsense()
    # api = OpenCV()
    node = CameraNode(api)

    bridge = CvBridge()
    # if 'intrinsic' in node.meta:
    #     with open('intrinsic.npy', 'wb') as f:
    #         np.save(f, node.meta['intrinsic'])

    CameraInfoPublisher(node, node.meta['intrinsic'], 'realsense')

    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    for data in node.api.capture():
        rgb = data['rgb']
        msg =  bridge.cv2_to_imgmsg(rgb, encoding=f'8UC{rgb.shape[-1]}')
        node.publisher.publish(msg)

        if 'depth' in data:
            data['depth'] = data['depth'][..., None]

        msg = bridge.cv2_to_imgmsg(data['depth'], encoding=f'16UC1')
        node.depth_publisher.publish(msg)
        # print(msg.height, msg.width, rgb.shape)
        if args.show:
            cv2.imshow('rgb', rgb[..., [2, 1, 0]])
            cv2.waitKey(1)

    if args.show:
        cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()
