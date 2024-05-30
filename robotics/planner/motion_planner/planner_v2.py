from typing import Sequence, Union
from mplib import Planner
from mplib.planner import ta
import numpy as np
from numpy import ndarray

from mplib.pymp import *

class PlannerV2(Planner):
    def __init__(self, urdf: str, user_link_names: Sequence[str], user_joint_names: Sequence[str], move_group: str, joint_vel_limits: Union[Sequence[float], ndarray], joint_acc_limits: Union[Sequence[float], ndarray], srdf: str = "", package_keyword_replacement: str = ""):
        super().__init__(urdf, user_link_names, user_joint_names, move_group, joint_vel_limits, joint_acc_limits, srdf, package_keyword_replacement)

        #self.control_based_planner = control_based.ControlBasedPlanner(self.planning_world)
        try:
            self.dubins_planner = dubins.DubinsPlanner(self.planning_world)
        except Exception as e:
            print("Dubins not loaded ..")
            self.dubins_planner = None


    def plan_control_based(
        self,
        goal_pose,
        current_qpos,
        mask = [],
        time_step=1., #0.1,
        rrt_range=0.1,
        planning_time=1.,
        fix_joint_limits=True,
        use_point_cloud=False,
        use_attach=False,
        verbose=False,
        integration_step=0.01,
    ):
        self.planning_world.set_use_point_cloud(use_point_cloud)
        self.planning_world.set_use_attach(use_attach)
        n = current_qpos.shape[0]
        if fix_joint_limits:
            for i in range(n):
                if current_qpos[i] < self.joint_limits[i][0]:
                    current_qpos[i] = self.joint_limits[i][0] + 1e-3
                if current_qpos[i] > self.joint_limits[i][1]:
                    current_qpos[i] = self.joint_limits[i][1] - 1e-3


        self.robot.set_qpos(current_qpos, True)
        collisions = self.planning_world.collide_full()
        if len(collisions) != 0:
            print("Invalid start state!")
            for collision in collisions:
                print("%s and %s collide!" % (collision.link_name1, collision.link_name2))

        idx = self.move_group_joint_indices
        ik_status, goal_qpos = self.IK(goal_pose, current_qpos, mask)
        if ik_status != "Success":
            return {"status": ik_status}

        if verbose:
            print("IK results:")
            for i in range(len(goal_qpos)):
               print(goal_qpos[i])

        goal_qpos_ = []
        for i in range(len(goal_qpos)):
            goal_qpos_.append(goal_qpos[i][idx])
        self.robot.set_qpos(current_qpos, True)
        
        status, path = self.dubins_planner.plan(
            current_qpos[idx],
            goal_qpos_, 
            range=rrt_range,
            verbose=verbose,
            time=planning_time,
            # integration_step=integration_step,
        )

        if status == "Exact solution": # or status == "Approximate solution":
            if verbose:
                ta.setup_logging("INFO")
            else:
                ta.setup_logging("WARNING")
            if status == 'Approximate solution':
                print("Got approximate solution!")
                print(path[-1])

            times, pos, vel, acc, duration = self.TOPP(path, time_step)
            return {
                "status": "Success",
                "time": times,
                "position": pos,
                "velocity": vel,
                "acceleration": acc,
                "duration": duration,
            }
        else:
            return {"status": "RRT Failed. %s" % status}