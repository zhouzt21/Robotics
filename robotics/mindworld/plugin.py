from typing import Optional, Union, List, Dict, Tuple, TYPE_CHECKING
import time
import rclpy
import numpy as np
import threading

import sapien.render
import rclpy.time
from sensor_msgs.msg import CameraInfo
from .mind import Mind

from robotics.utils.camera_utils import RGBDepthSubscriber, rgb_depth2pointcloud
from robotics import Pose


class Plugin:
    def setup(self, mind: Mind):
        self.mind = mind
        self.node = mind.node

    def before_render_step(self):
        pass

        
        
class RGBDVisualizer(Plugin):
    def __init__(self, scales=0.005, opacity=0., max_depth=10.0, N=0, rate=1.) -> None:
        super().__init__()
        self.scales = scales
        self.opacity = opacity
        self.max_depth = max_depth
        self.N = N
        self.rate = rate

    def setup(self, mind: Mind):
        super().setup(mind)

        self.actor = None
        self._last = time.time()


        self.node.listen_once(CameraInfo, '/camera_info', self.camera_callback, is_static=True)
        self.data = None

        self.lock = threading.Lock()

        def callback(*msg):
            #data = self.subscription.extract_rgb_depth(msg)
            with self.lock:
                self.data = msg

        self.subscription = RGBDepthSubscriber(self.node, callback)

    def camera_callback(self, msg: CameraInfo):
        self.intrinsic = np.array(msg.k).reshape((3, 3))
        self.frame_name = msg.header.frame_id

    def before_render_step(self):
        #return super().before_render_step()
        with self.lock:
            data = self.data
            if data is not None:
                self.data = None

        if data is None:
            return

        if self.actor is not None and time.time() - self._last > 1. / self.rate:
            self.mind.scene.remove_actor(self.actor)
            self.actor = None

        
        if self.actor is None:
            data = self.subscription.extract_rgb_depth(data)
            image = data['rgb']
            depth = data['depth']

            xyz, rgb = rgb_depth2pointcloud(image, depth, self.intrinsic, max_depth=self.max_depth)
            rgb = rgb.astype(np.float64) / 255.

            if len(xyz) == 0:
                return


            if self.N and len(xyz) > self.N:
                idx = np.random.choice(len(xyz), self.N, replace=False)
                xyz = xyz[idx]
                rgb = rgb[idx]


            pose = self.mind.robot.lookup_transform('base_link', self.frame_name)
            if pose is not None:
                self.actor, _ = self.mind.add_pointcloud(xyz, rgb, self.scales, 0)
                self._last = time.time()
                pose = self.mind.get_robot_frame() * pose
                self.actor.set_pose(pose)
