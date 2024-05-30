from typing import Any, Optional, Dict
from ..skill import Skill, SkillConfig
from ..agent import Agent


class SequentialConfig(SkillConfig):
    pass

class Sequential(Skill):
    config: SequentialConfig

    def __init__(self, *args: Skill, config: Optional[SkillConfig]=None) -> None:
        config = config or SequentialConfig()
        super().__init__(config)
        self.n = len(args)
        self.subskills = list(args)

    def reset(self, agent: Agent, obs: Dict, **kwargs):
        super().reset(agent, obs, **kwargs)
        self.skill_idx = 0
        print("Switching to skill", self.skill_idx, str(self.subskills[self.skill_idx]))
        self.subskills[0].reset(agent, obs, **kwargs)

    def act(self, obs, **kwargs) -> Any:
        return self.subskills[self.skill_idx].act(obs, **kwargs)

    def post_act(self):
        super().post_act()
        self.subskills[self.skill_idx].post_act()

    def should_terminate(self, obs, **kwargs):
        while self.skill_idx < self.n:
            s = self.subskills[self.skill_idx]
            if s.should_terminate(obs, **kwargs) or s.is_timeout():
                self.skill_idx += 1
                if self.skill_idx < self.n:
                    print("Switching to skill", self.skill_idx, str(self.subskills[self.skill_idx]))
                    self.subskills[self.skill_idx].reset(self.agent, obs, **kwargs)
            else:
                break

        return self.skill_idx >= self.n