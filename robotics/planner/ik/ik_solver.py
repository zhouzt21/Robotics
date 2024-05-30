import os
import sapien.physx
from sapien.physx import PhysxRigidBodyComponent
import transforms3d
import numpy as np
from mplib import Planner
from typing import Optional, cast, Tuple
from robotics.sim import Simulator
from robotics import Pose
from sapien.utils.viewer import Viewer


from robotics.mindworld.draw_utils import create_coordinate_axes, set_coordinate_axes
from robotics.utils.sapien_utils import get_rigid_dynamic_component

class IKSolver:
    def __init__(
        self, sim: Simulator, ee_name: str,
    ) -> None:
        self.sim = sim
        robot = sim.robot
        assert robot is not None
        self.robot = robot


        loader = sim._scene.create_urdf_loader()
        self.fake_articulation = loader.load(robot.urdf_path, robot.srdf)
        import sapien.render
        for i in self.fake_articulation.get_links():
            for j in i.entity.components:
                if isinstance(j, sapien.render.RenderBodyComponent):
                    j.visibility = 0.1
            i.disable_gravity = True

        for i in self.fake_articulation.get_active_joints():
            i.set_drive_properties(0., 0.)
        

        self.articulation = self.robot.articulation
        self.pmodel = robot.articulation.create_pinocchio_model() # type: ignore

        link_names = [i.name for i in robot.articulation.get_links()]
        joint_names = [i.name for i in robot.articulation.get_active_joints()]

        #vel_limits = 
        self.arm_indices = joint_indices = robot.controllers['arm'].joint_indices
        if 'base' in robot.controllers:
            self.base_indices = robot.controllers['base'].joint_indices
            joint_indices = np.concatenate([joint_indices, robot.controllers['base'].joint_indices])
        else:
            self.base_indices = []

        self.qmask = np.zeros(self.articulation.dof, dtype=bool)
        ee_link = None
        for i in self.articulation.get_links():
            if i.name == ee_name:
                ee_link = i

        assert ee_link is not None, f'cannot find ee link {ee_name}'
        self.ee_link = ee_link
        self.ee_link_idx = self.articulation.get_links().index(self.ee_link)

        self.joint_limits = np.array([i.get_limits() for i in self.articulation.get_active_joints()])[:, 0]
        self.qpos = np.array(self.articulation.get_qpos()).astype(np.float64)

        self.ee_target: Optional[Pose] = None
        self.base_inv = cast(Pose, self.articulation.get_pose().inv())



        # planner with fixed base
        urdf_path = robot.urdf_path
        srdf_path = robot.get_srdf_path()
        if len(self.base_indices) > 0:
            # HACK: modify urdf manually
            from xml.etree import ElementTree as ET
            from xml.etree.ElementTree import Element

            tree = ET.parse(robot.urdf_path)
            root: Element = tree.getroot()
            _joint_names = [i.name for i in self.fake_articulation.get_active_joints()]
            base_joint_names = [_joint_names[i] for i in self.base_indices]

            joint_names = [i.name for i in self.fake_articulation.get_active_joints() if i.name not in base_joint_names]
            for i in root.iter('joint'):
                if i.attrib['name'] in base_joint_names: 
                    i.attrib['type'] = 'fixed'
                    print(i.attrib['name'], 'fixed')


            urdf_path = urdf_path.split('.')[0] + '_fixed_base.urdf'
            tree.write(urdf_path)
            if srdf_path is not None and srdf_path != "":
                new_srdf_path = srdf_path.split('.')[0] + '_fixed_base.srdf'
                os.system(f'cp {srdf_path} {new_srdf_path}')
                srdf_path = new_srdf_path


        vel_limits = np.ones(len(self.arm_indices))
        self.planner = Planner(
            urdf=urdf_path,
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group = ee_name,
            srdf = srdf_path,
            joint_vel_limits=vel_limits,
            joint_acc_limits=np.ones_like(vel_limits)
        )
        print("initialized planner")


        self.ee_target_input = None
        self.qpos_input = None
        self.move_base = False
       
        self.last_ee = None 
        self.last_qpos = None
        self.arm_action = None

        
    def check_collision(self, qpos, verbose):
        self.planner.robot.set_qpos(qpos, True)
        collisions = self.planner.planning_world.collide_full()
        if len(collisions) != 0 and verbose:
            print("Invalid start state!")
            for collision in collisions:
                print("%s and %s collide!" % (collision.link_name1, collision.link_name2))
        return collisions


    def distance_6D(self, p1, q1, p2, q2):
        return np.linalg.norm(p1 - p2) + min(
            np.linalg.norm(q1 - q2), np.linalg.norm(q1 + q2) # type: ignore
        )


    def no_self_collision(self, qpos):
        self.planner.planning_world.set_qpos_all(qpos[self.planner.move_group_joint_indices])
        return len(self.planner.planning_world.collide_full()) == 0

    def add_scene_collision(self, pcd=None):
        if pcd is None:
            pcd = self.sim.gen_scene_pcd()
        self.planner.update_point_cloud(pcd)
        self.planner.planning_world.set_use_point_cloud(True)


    def mplib_IK(self, planner: Planner, goal_pose, start_qpos, mask = [], n_init_qpos=20, threshold=1e-3, base_sampler=None):
        #index = planner.link_name_2_idx[planner.move_group]
        index = self.ee_link_idx
        min_dis = 1e9
        # idx = planner.move_group_joint_indices
        qpos0 = np.copy(start_qpos)
        results = []
        idx = self.base_indices + self.arm_indices

        goal_pose_sapien = Pose(goal_pose[:3], goal_pose[3:])
        #print(start_qpos)
        #print(goal_pose)
        for i in range(n_init_qpos):
            ik_results = self.pmodel.compute_inverse_kinematics(
                index, goal_pose_sapien, start_qpos, np.logical_not(mask), max_iterations = 4000
            )
            ik0 = ik_results[0]
            for j in range(3, 9):
                val = ik0[j]
                while val > np.pi:
                    val = val - np.pi * 2
                while val < -np.pi:
                    val = val + np.pi * 2
                ik0[j] = val

            # NOTE: assuming all joints are bounded
            flag = (ik0[3:9] > self.joint_limits[3:9, 0]).all() and (ik0[3:9] < self.joint_limits[3:9, 1]).all()
            flag = flag and self.no_self_collision(ik0[3:])

            if flag:
                #planner.pinocchio_model.compute_forward_kinematics(ik_results[0])
                #new_pose = planner.pinocchio_model.get_link_pose(index)
                self.pmodel.compute_forward_kinematics(ik0)
                new_pose = self.pmodel.get_link_pose(index)
                tmp_dis = self.distance_6D(
                    goal_pose[:3], goal_pose[3:], new_pose.p, new_pose.q
                )
                if tmp_dis < min_dis:
                    min_dis = tmp_dis
                if tmp_dis < threshold:
                    result = ik_results[0] 
                    unique = True
                    for j in range(len(results)):
                        if np.linalg.norm(results[j][idx] - result[idx]) < 0.1: # type: ignore
                            unique = False
                    if unique:
                        results.append(result)
            # start_qpos = self.pmodel.get_random_qpos()

            start_qpos = planner.pinocchio_model.get_random_configuration()
            if len(self.base_indices) > 0:
                if base_sampler is None:
                    assert len(start_qpos) + len(self.base_indices) == len(qpos0)

                    q = np.random.random((2,)) * 2  - 1 + qpos0[:2]
                    theta = np.random.random((1,)) * 2 * np.pi - np.pi
                else:
                    q, theta = base_sampler()
                start_qpos = np.concatenate([q, theta, start_qpos])

            if len(mask) > 0:
                start_qpos[mask] = qpos0[mask]
        if len(results) != 0:
            status = "Success"
        elif min_dis != 1e9:
            status = (
                "IK Failed! Distance %lf is greater than threshold %lf."
                % (min_dis, threshold)
            )
        else:
            status = "IK Failed! Cannot find valid solution."
        return status, results

    @staticmethod
    def plan(
        planner: Planner,
        goal_qpos,
        current_qpos,
        time_step=0.1,
        rrt_range=0.1,
        planning_time=1,
        use_point_cloud=False,
        use_attach=False,
        verbose=False,
        planner_name="RRTConnect"
    ):
        import toppra as ta
        planner.planning_world.set_use_point_cloud(use_point_cloud)
        planner.planning_world.set_use_attach(use_attach)
        print(goal_qpos)
        print(current_qpos)

        planner.robot.set_qpos(current_qpos, True)
        
        status, path = planner.planner.plan(
            current_qpos[planner.move_group_joint_indices],
            goal_qpos, 
            range=rrt_range,
            verbose=verbose,
            time=planning_time,
            planner_name=planner_name
        )

        if status == "Exact solution":
            if verbose:
                ta.setup_logging("INFO")
            else:
                ta.setup_logging("WARNING")

            times, pos, vel, acc, duration = planner.TOPP(path, time_step)
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


    def IK(self, goal_pose: Pose, start_qpos: np.ndarray, fix_base=True, n_init_qpos=20, base_sampler=None):
        mask = self.qmask.copy()
        if fix_base:
            mask[self.base_indices] = True
        goal_pose_list = list(goal_pose.p) + list(goal_pose.q)
        status, result = self.mplib_IK(self.planner, goal_pose_list, start_qpos, mask=mask, n_init_qpos=n_init_qpos, base_sampler=base_sampler)
        if len(result) > 0:
            result = result[0]
        else:
            result = None
        return status, result

    def visualize_ik(self, ik_status, ik_qpos, move_base):

        if ik_qpos is not None:
            self.fake_articulation.set_qpos(ik_qpos)

        if ik_status == "Success":
            if move_base:
                sol = ik_qpos # type: ignore
                self.base_axis.set_scale([0.1] * 3) # type: ignore
                self.base_axis.set_position(np.append(sol[:2], 0))
                self.base_axis.set_rotation(transforms3d.euler.euler2quat(0, 0, sol[2]))

            self.block_material.base_color = [0, 1, 0, 1]
        else:
            self.block_material.base_color = [1, 0, 0, 1]

    
    def ik_move_base(self, grasp_pose: Pose, radius: Tuple[float, float]=(0.1, 0.3)):
        # sample base in a circle towards the grasp pose
        self.block.set_pose(grasp_pose)
        set_coordinate_axes(self.axis, grasp_pose)

        def base_sampler():
            theta = np.random.random((1,)) * 2 * np.pi
            r = np.random.random((1,)) * radius
            q = grasp_pose.p[:2] - np.array([np.cos(theta[0]), np.sin(theta[0])]) * r[0] # type: ignore
            return q, theta
        status, qpos = self.IK(grasp_pose, self.qpos, fix_base=False, n_init_qpos=20, base_sampler=base_sampler)

        self.visualize_ik(status, qpos, move_base=True)
        return status, qpos


    def setup_sapien(self):
        sim = self.sim
        builder = sim._scene.create_actor_builder()
        self.block_material = sim._renderer.create_material()
        self.block_material.base_color = [1., 0., 0., 1.]
        builder.add_box_visual(Pose((0, 0, 0.)), (0.02, 0.02, 0.02), self.block_material)
        self.block = builder.build_kinematic()

        self.axis = create_coordinate_axes(sim)

        self.base_axis = create_coordinate_axes(sim)
        for c in self.base_axis.children:
            c.transparency = 0.5 # type: ignore
        self.base_axis.set_scale([0.1] * 3) # type: ignore