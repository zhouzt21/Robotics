import gymnasium as gym
import numpy as np

import sapien.core as sapien
from sapien.core import Pose
from robotics.sim import RobotBase, Simulator
from typing import Dict, Union, List, TYPE_CHECKING, cast
from ..sim import Simulator, RobotBase, MobileSmallRobot


from robotics.sim.robot.mobile_small_v2 import MobileSmallV2


if TYPE_CHECKING:
    from .skill import Skill


class Agent:
    skills: List["Skill"]

    def __init__(self) -> None:
        self.skills = []
        self._obs = None

        self.attached_objects = ()
        self.gripper_target = 0.

    def load(self, sim: Simulator, robot: RobotBase):
        self.simulator = sim
        self.robot = robot
        self.action_space = robot.action_space

    def gen_scene_pcd(self):
        return self.simulator.gen_scene_pcd(exclude=self.attached_objects)

    def gen_attach_mesh(self):
        if len(self.attached_objects) > 0:
            from robotics.utils.sapien_utils import get_actor_mesh
            assert len(self.attached_objects) == 1
            actor_id = self.attached_objects[0]

            actors = self.simulator._scene.get_all_actors()
            actor = None
            for i in actors:
                if i.get_name() == actor_id:
                    actor = i
                    break
            assert actor is not None

            robot = self.robot
            links = robot.get_articulations()[0].get_links()
            link = None
            for i in links:
                if i.get_name() == robot.ee_name:
                    link = i
                    break
            assert link is not None


            mesh = get_actor_mesh(actor)
            if mesh:
                matrix = cast(np.ndarray, link.get_pose().inv().to_transformation_matrix())
                mesh.apply_transform(matrix)
            return mesh
        return None


    def attach(self, actor_id):
        assert len(self.attached_objects) == 0, "Only support one attached object"
        print("attached..", actor_id)
        self.attached_objects = (actor_id,)

    def detach_all(self):
        self.attached_objects = ()

    def get_observation(self):
        output = {}
        for i in self.skills:
            key = i.get_key()
            assert key not in output, f"Duplicate skill {key} in the observation"
            output[key] = i.get_observation(self._obs)
        return output
    
    def act(self, obs: Dict, *skills: "Skill"):
        """
        act on the observation
        we always assume observation is a dict
        """
        self._obs = obs
        for i in skills:
            i.reset(self, obs)
            self.skills.append(i)

        obs['_skills'] = self.get_observation()

        for i in self.skills:
            if i.should_terminate(obs) or i.is_timeout():
                i._terminated = True
                self.skills.remove(i)
                print("Skill", i, "terminated")
                i.close()

        actions = []
        for i in self.skills:
            actions.append(i.act(obs))
            i.post_act()

        return self.resolve(actions)

    def resolve(self, actions):
        raise NotImplementedError


class MobileAgent(Agent):
    action_space: gym.spaces.Box
    robot: MobileSmallRobot

    def load(self, sim: Simulator, robot: MobileSmallRobot):
        super().load(sim, robot)
        self.indices = {}
        for k, v in robot.controllers.items():
            self.indices[k] = v.joint_indices
        self.action_map = robot._action_mapping

        self.motion_model = robot.motion_model

        self.articulated: sapien.physx.PhysxArticulation = robot.get_articulations()[0]
        self.pinocchio_model = self.articulated.create_pinocchio_model() # type: ignore
        self.move_group_joints = [
            j.name
            for j in self.articulated.get_joints()
            if j.get_dof() != 0
        ]

        self.link_idx = [i.name for i in self.articulated.get_links()].index(robot.ee_name)
        self.ee = self.articulated.get_links()[self.link_idx]

    def ee_pose(self):
        return self.ee.get_pose()

    def set_base_move(self, actions):
        s, t = self.action_map['base']
        if t - s == 2:
            actions = actions[[0, 2]]
        def set_base(action):
            action[s:t] = actions
            return action
        return set_base

    def compute_ik(self, pose: Pose):
        # TODO: new IK version
        raise NotImplementedError("TODO: new IK version")
        mask = np.array(
            [False for j in self.move_group_joints]
        ).astype(int)
        mask[self._arm] = True

        pose = self.articulated.pose.inv() * self.robot.base_pose * pose
        result, success, error = self.pinocchio_model.compute_inverse_kinematics(
            self.link_idx,
            pose,
            initial_qpos=self.articulated.get_qpos(), # type: ignore
            active_qmask=mask, # type: ignore
            max_iterations=100,
        )
        return result, success, error


    def resolve(self, actions):
        action = np.zeros(self.action_space.shape)


        if 'gripper' in self.indices:
            s, e = self.indices['gripper']
            action[s:e] = self.robot.controllers['gripper'].drive_target(self.gripper_target)

        for items in actions:
            action = items(action)
        return action


    def get_qpos(self):
        return self.articulated.get_qpos()