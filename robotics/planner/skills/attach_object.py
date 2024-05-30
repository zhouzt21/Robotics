# change the state of the objects ...
from typing import Any, Dict

from robotics.planner.agent import Agent
from ..skill import Skill, SkillConfig


class AttachObject(Skill):
    def __init__(self, actor_id) -> None:
        super().__init__(SkillConfig())
        self.actor_id = actor_id

    def reset(self, agent: Agent, obs: Dict, **kwargs):
        super().reset(agent, obs, **kwargs)
        agent.attach(self.actor_id)

    def act(self, obs, **kwargs) -> Any:
        return [], []

    def should_terminate(self, obs, **kwargs):
        # terminated immediately
        return True