from typing import Any, Optional, Dict
from ..skill import Skill, SkillConfig
from ..agent import Agent


class ComposeConifg(SkillConfig):
    pass

class Compose(Skill):
    config: ComposeConifg

    def __init__(self, *args: Skill, config: Optional[ComposeConifg]=None) -> None:
        config = config or ComposeConifg()
        super().__init__(config)
        self.n = len(args)
        self.skills = list(args)

    def reset(self, agent: Agent, obs: Dict, **kwargs):
        super().reset(agent, obs, **kwargs)
        for i in range(self.n):
            self.skills[i].reset(agent, obs, **kwargs)

    def act(self, obs, **kwargs) -> Any:
        outs = []
        for i in self.skills:
            outs.append(i.act(obs, **kwargs))

        def f(action):
            for i in outs:
                action = i(action)
            return action
        return f

    def post_act(self):
        super().post_act()
        for i in self.skills:
            i.post_act()

    def should_terminate(self, obs, **kwargs):
        for i in self.skills:
            if i.should_terminate(obs) or i.is_timeout():
                i._terminated = True
                self.skills.remove(i)
                print("Skill", i, "terminated")
                i.close()
        return len(self.skills) == 0