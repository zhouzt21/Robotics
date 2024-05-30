from robotics.planner.agent import Agent
import cv2
import numpy as np
from ..skill import Skill, SkillConfig
from typing import Any, Dict, Optional, Union, List

from robotics.utils import logger


class ShowCameraConfig(SkillConfig):
    cameras: Union[str, List[str]] = ''
    render_mode: Union[str, List[str]] = 'rgb'
    freq: int = 4
    topic: str = 'camera'
    use_cv2: bool = True

    
def parse_str_list(x: Union[str, List[str]]) -> List[str]:
    if isinstance(x, str):
        return [x]
    return x


class ShowCamera(Skill):
    config: ShowCameraConfig

    def __init__(self, config: ShowCameraConfig) -> None:
        super().__init__(config)

        self.camera_names = parse_str_list(config.cameras)
        self.render_mode = parse_str_list(config.render_mode)

        
    def reset(self, agent: Agent, obs: Dict, **kwargs):
        super().reset(agent, obs, **kwargs)

        element = agent.simulator.elements
        self.cameras = [element.find(name) for name in self.camera_names]

    
    def act(self, obs, **kwargs) -> Any:
        self.render()
        return lambda x: x

    def should_terminate(self, obs, **kwargs):
        return self._terminated
    
    def render(self):
        if self.agent.simulator._viewer is None:
            return
        if self._elapsed_steps % self.config.freq != 0:
            return
        #for camera, mode in zip(self.cameras, self.render_mode):
        #    camera.render(mode=mode)
        all_cameras = []

        for camera in self.cameras:
            assert camera is not None
            images = []
            for mode in self.render_mode:
                obs = camera.get_observation()
                if mode == 'depth':
                    position = obs["Position"]
                    gray = position[..., 2]
                    gray = (gray - gray.min()) / (gray.max() - gray.min())
                    rgb = (cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) * 255).astype(np.uint8)
                    images.append(rgb)
                elif mode == 'rgb':
                    images.append((obs["Color"] * 255).astype(np.uint8)[..., [0, 1, 2]])
                else:
                    raise NotImplementedError
            all_cameras.append(np.concatenate(images, axis=0))
        
        img = np.concatenate(all_cameras, axis=1)

        logger.log(self.config.topic, img, 2)

        if self.config.use_cv2:
            cv2.imshow("show_camera", img[..., [2, 1, 0]])
            key = cv2.waitKey(1)
            if key == ord('q'):
                self._terminated = True
        
            
def show_camera(**kwargs):
    return ShowCamera(ShowCameraConfig(**kwargs))