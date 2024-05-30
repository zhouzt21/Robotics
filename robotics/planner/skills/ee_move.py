from typing import Any, Dict, Sequence, Tuple
from ..agent import MobileAgent
from ..skill import Skill, SkillConfig, SkillExecutionError
#from realbot.sim.cfg import PoseConfig
import numpy as np
#from realbot.agent.motion_planner.planner_v2 import PlannerV2
from mplib import Planner as PlannerV2
from robotics.utils import logger


class EEMoveConfig(SkillConfig):
    #target: PoseConfig = PoseConfig(p=(0.5, -0.5, 0.))
    target_p: Tuple[float, ...] = (0.5, -0.5, 0.)
    target_q: Tuple[float, ...] = (1., 0, 0, 0)

    planning_time: float = 2.
    use_attach: bool = True
    close_gripper: bool = True

    control_steps: int = 1


class EEMove(Skill):
    config: EEMoveConfig
    planner: PlannerV2
    agent: MobileAgent

    def reset(self, agent: MobileAgent, obs: Dict, **kwargs):
        super().reset(agent, obs, **kwargs)

        # self.driver = Driver(agent)

        robot = agent.articulated
        if not hasattr(agent, 'planner'):
            urdf_path = agent.robot.get_urdf_path()
            link_names = [link.get_name() for link in robot.get_links()]
            joint_names = [joint.get_name() for joint in robot.get_active_joints()]

            planner = PlannerV2(
                urdf=urdf_path,
                user_link_names=link_names,
                user_joint_names=joint_names,
                move_group = agent.ee.name,
                srdf = "",
                joint_vel_limits=np.ones(7 + 3),
                joint_acc_limits=np.ones(7 + 3)
            )
            setattr(agent, 'planner', planner)

        self.planner = getattr(agent, 'planner')
        pose = self.config.target_p + self.config.target_q


        attachment = self.agent.gen_attach_mesh()
        if attachment is not None:
            attachment.export('/tmp/ee_move_mesh.stl')
            self.planner.update_attached_mesh('/tmp/ee_move_mesh.stl', [0, 0, 0, 1, 0, 0, 0])

        pcd = self.agent.gen_scene_pcd()
        if pcd is not None:
            self.planner.update_point_cloud(pcd)

        use_attach = self.config.use_attach and attachment is not None
        print("Attaching ..", use_attach, 'pcd', pcd is not None)

        if self.agent.motion_model == 'mobile':
            #TODO: plan with fixing the base ..
            self.result = self.planner.plan_control_based(
                pose, robot.get_qpos(), time_step=1/250, 
                planning_time=20., #self.config.planning_time + 30., 
                use_point_cloud=pcd is not None, 
                use_attach=use_attach,
                verbose=True,
                rrt_range=0.1,
                integration_step=0.01, 
            )

        else:
            self.result = self.planner.plan_screw(
                pose, robot.get_qpos(), time_step=1/250,
                use_point_cloud=pcd is not None, 
                use_attach=use_attach,
            )
            if self.result['status'] != "Success":
                print("screw motion failued")
                self.result = self.planner.plan(
                    pose, robot.get_qpos(), time_step=1/250, 
                    planning_time=self.config.planning_time, 
                    use_point_cloud=pcd is not None, 
                    use_attach=use_attach, 
                    verbose=True
                )

        if not self.result['status'] == 'Success':
           self._terminated = True
           raise SkillExecutionError("Planning failed")

    
    def act(self, obs, **kwargs) -> Any:
        assert not self._terminated
        qpos = self.agent.get_qpos()
        pos = self.result['position'][self._elapsed_steps//self.config.control_steps]

        actions = {}
        for k, v in self.agent.indices.items():
            if k != 'gripper':
                actions[k] = self.agent.robot.controllers[k].drive_target(pos[v])

        logger.log("qpos", qpos)
        logger.log("target", pos)

        def set_action(action):
            for k, v in self.agent.action_map.items():
                if k!= 'gripper':
                    start, end = v
                    action[start:end] = actions[k]
            return action
        return set_action

    def should_terminate(self, obs, **kwargs):
        return self._terminated or (self._elapsed_steps // self.config.control_steps) >= len(self.result['position'])

        
        
def move_to_pose(target, **kwargs):
    if isinstance(target, list):
        p = float(target[0]), float(target[1]), float(target[2])
        q = float(target[3]), float(target[4]), float(target[5]), float(target[6])
        #target = PoseConfig(p=p, q=q)
    return EEMove(EEMoveConfig(target_p=p, target_q=q, **kwargs))