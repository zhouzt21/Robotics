from typing import Any, Dict
from robotics.planner.agent import MobileAgent
from ..skill import Skill, SkillConfig

class CloseConfig(SkillConfig):
    timeout: int = 20

class GripperSkill(Skill):
    target_qpos: float
    config: CloseConfig
    agent: MobileAgent

    def reset(self, agent: MobileAgent, obs: Dict, **kwargs):
        super().reset(agent, obs, **kwargs)
        self.agent.gripper_target = self.target_qpos

    def act(self, obs, **kwargs) -> Any:
        return lambda x: x

    def should_terminate(self, obs, **kwargs):
        return False



class CloseGripper(GripperSkill):
    target_qpos: float = 0.0

    
class OpenGripper(GripperSkill):
    target_qpos: float = 0.4

    
def close_gripper():
    return CloseGripper(CloseConfig())

def open_gripper():
    return OpenGripper(CloseConfig())