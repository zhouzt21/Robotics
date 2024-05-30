from robotics.utils.config import Config
from typing import Dict, Any
from .agent import Agent


class SkillExecutionError(Exception):
    pass


class SkillConfig(Config):
    timeout: int = 0
    name: str = ""


class Skill:
    config: SkillConfig
    _terminated: bool = False

    def __str__(self) -> str:
        return self.config.name or super().__str__()

    def get_key(self):
        return self.__class__.__name__

    def __init__(self, config: SkillConfig) -> None:
        self.config = config

    def reset(self, agent: Agent, obs: Dict, **kwargs):
        self.agent = agent
        self._elapsed_steps = 0
        self._terminated = False

    def post_act(self):
        self._elapsed_steps += 1

    def is_timeout(self):
        timeout = self.config.timeout
        if timeout > 0:
            return self._elapsed_steps >= timeout
        else:
            return False

    def get_observation(self, obs, **kwargs):
        pass

    def get_state(self):
        raise NotImplementedError

    def set_state(self, state):
        raise NotImplementedError

    def act(self, obs, **kwargs) -> Any:
        # ensure that the agent should never be terminated
        raise NotImplementedError

    def should_terminate(self, obs, **kwargs):
        raise NotImplementedError

    def close(self):
        pass