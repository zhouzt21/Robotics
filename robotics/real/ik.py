"""motion planning for robot arm control 


We assume the robot arm supports a positional based control that supports positional-based control and avoids self-collision of the arm

The IKController will:

1. take an ee pose as input
2. compute the joint angles that can reach the ee pose
3. maintain a path that can reach the ee pose
4. return the joint angles that can reach the ee pose

"""
import os
import transforms3d
import numpy as np
from mplib import Planner
from typing import Optional, cast
from robotics.sim import Simulator
from robotics import Pose
from sapien.utils.viewer import Viewer

# try:
#     import pinnocchio as pin
# except ImportError:
#     raise ImportError('pinnocchio is required for motion planning, please run `pip install pin`')



def _create_coordinate_axes(viewer: Viewer):
    from sapien import internal_renderer as R
    renderer_context = viewer.renderer_context
    assert renderer_context is not None
    cone = renderer_context.create_cone_mesh(16)
    capsule = renderer_context.create_capsule_mesh(0.1, 0.5, 16, 4)
    mat_red = renderer_context.create_material(
        [1., 0, 0, 1], [0, 0, 0, 1], 0, 1, 0 # type: ignore
    ) 
    mat_green = renderer_context.create_material(
        [0, 1, 0, 1], [0, 0, 0, 1], 0, 1, 0 # type: ignore
    )
    mat_blue = renderer_context.create_material(
        [0, 0, 1, 1], [0, 0, 0, 1], 0, 1, 0 # type: ignore
    )
    red_cone = renderer_context.create_model([cone], [mat_red])
    green_cone = renderer_context.create_model([cone], [mat_green])
    blue_cone = renderer_context.create_model([cone], [mat_blue])
    red_capsule = renderer_context.create_model([capsule], [mat_red])
    green_capsule = renderer_context.create_model([capsule], [mat_green])
    blue_capsule = renderer_context.create_model([capsule], [mat_blue])

    assert viewer.system is not None
    render_scene: R.Scene = viewer.system._internal_scene

    def set_scale_position(obj, scale=None, position=None, rotation=None):
        if scale is not None:
            obj.set_scale(scale)
        if position is not None:
            obj.set_position(position)
        if rotation is not None:
            obj.set_rotation(rotation)
        obj.shading_mode = 0
        obj.cast_shadow = False
        obj.transparency = 1

    node = render_scene.add_node()
    obj = render_scene.add_object(red_cone, node)
    set_scale_position(obj, [0.5, 0.2, 0.2], [1, 0, 0])

    obj = render_scene.add_object(red_capsule, node)
    set_scale_position(obj, None, [0.5, 0, 0])

    obj = render_scene.add_object(green_cone, node)
    set_scale_position(obj, [0.5, 0.2, 0.2], [0, 1, 0], [0.7071068, 0, 0, 0.7071068])

    obj = render_scene.add_object(green_capsule, node)
    set_scale_position(obj, None, [0, 0.5, 0], [0.7071068, 0, 0, 0.7071068])

    obj = render_scene.add_object(blue_cone, node)
    set_scale_position(obj, [0.5, 0.2, 0.2], [0, 0, 1], [0, 0.7071068, 0, 0.7071068])

    obj = render_scene.add_object(blue_capsule, node)
    set_scale_position(obj, None, [0, 0, 0.5], [0, 0.7071068, 0, 0.7071068])

    return node



class PathCache:
    """_summary_
    store the intermediate path
    """
    def __init__(self, path, planner: Planner) -> None:
        self.path = path
        self.planner = planner

    def no_collision(self, a):
        self.planner.robot.set_qpos(a, False)
        return len(self.planner.planning_world.collide_full()) == 0

    def can_connect(self, a: np.ndarray, b: np.ndarray, eps: float=0.03):
        n = int(np.ceil(np.abs(b - a).max() / eps))
        if n <= 1:
            return True

        step = (b - a) / n
        for i in range(n):
            if not self.no_collision(a + step * (i+1)):
                return False
        return True

    def connect(self, path, qpos, eps):
        out = -1
        for i in range(len(path)):
            if self.can_connect(qpos, path[i], eps):
                out = i
        return out

    def update(self, qpos: np.ndarray, target_qpos: np.ndarray, eps: float = 0.03) -> bool:
        """_summary_
        find the nearest path and check if they can be connected
        """
        idx = self.connect(self.path, qpos, eps)
        if idx != -1:
            self.path = self.path[idx:]
        else:
            return False

        idx = self.connect(self.path[::-1], target_qpos, eps)
        if idx != -1:
            self.path = self.path[:len(self.path)-idx] + [target_qpos]
        else:
            return False
        return True



class IKSolver:
    def __init__(
        self, 
        sim: Simulator,
        ee_name: str,
        ee_bounds: Optional[np.ndarray]=None,
    ) -> None:
        self.sim = sim
        robot = self.robot = sim.robot
        self.ee_bounds = ee_bounds


        self.fake_articulation = robot.loader.load(robot.urdf_path, robot.srdf)
        import sapien.render
        for i in self.fake_articulation.get_links():
            for j in i.entity.components:
                if isinstance(j, sapien.render.RenderBodyComponent):
                    j.visibility = 0.1
        

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
        self.planner =  Planner(
            urdf=urdf_path,
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group = ee_name,
            srdf = srdf_path,
            joint_vel_limits=vel_limits,
            joint_acc_limits=np.ones_like(vel_limits)
        )
        print("initialized planner")


        
        self.path_cache = None

        
        import threading

        self.input_lock = threading.Lock()
        self.output_loc = threading.Lock()
        self.plan_thread = threading.Thread(target=self.planning_loop, daemon=True)


        self.ee_target_input = None
        self.qpos_input = None
        self.move_base = False
       
        self.last_ee = None 
        self.last_qpos = None
        self.arm_action = None

        self.plan_thread.start()
        # self.ee_target = None

    def send_ee_pose(self, ee_pose, cur_qpos, move_base):
        self.block.set_pose(ee_pose)
        self.show_axis(ee_pose)


        with self.input_lock:
            self.ee_target_input = ee_pose
            self.qpos_input = cur_qpos
            self.move_base = move_base
        
        
    def planning_loop(self):
        """_summary_
        TODO: 
            - the whole IK server is run on another thread
            - has a single lock to read and write the certain variable shared with the main thread
                input:
                    - ee_target to solve
                    - the recent qpos 
                output:
                    - last ee found 
                    - found ik qpos
        """
        import copy
        with self.input_lock:
            target_ee = copy.deepcopy(self.ee_target_input)
            cur_q = copy.deepcopy(self.qpos_input)
            move_base = self.move_base
        
        #while True:
        #    self.step(target_ee, cur_q, move_base, self.last_ee, self.last_qpos)

        
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


    def mplib_IK(self, planner: Planner, goal_pose, start_qpos, mask = [], n_init_qpos=20, threshold=1e-3):
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
                index, goal_pose_sapien, start_qpos, np.logical_not(mask)
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
            # if not flag:
            #     print('out of limit', flag, ik0[3:9], self.joint_limits[3:9, 0], self.joint_limits[3:9, 1])
            #     exit(0)
            # no_collision = self.no_self_collision(ik0[3:])
            # if not no_collision:
            #     for col in self.planner.planning_world.collide_full():
            #         print(col.link_name1, col.link_name2)
            # print(flag, no_collision)
            flag = flag and self.no_self_collision(ik0[3:])
            # print('failed', flag)

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
                        if np.linalg.norm(results[j][idx] - result[idx]) < 0.1:
                            unique = False
                    if unique:
                        results.append(result)
            # start_qpos = self.pmodel.get_random_qpos()

            start_qpos = planner.pinocchio_model.get_random_configuration()
            if len(self.base_indices) > 0:
                assert len(start_qpos) + len(self.base_indices) == len(qpos0)

                q = np.random.random((2,)) * 2  - 1 + qpos0[:2]
                theta = np.random.random((1,)) * 2 * np.pi - np.pi
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


    def IK(self, goal_pose: Pose, start_qpos: np.ndarray, fix_base=True, n_init_qpos=20):

        mask = self.qmask.copy()
        if fix_base:
            mask[self.base_indices] = True
        goal_pose_list = list(goal_pose.p) + list(goal_pose.q)
        status, result = self.mplib_IK(self.planner, goal_pose_list, start_qpos, mask=mask, n_init_qpos=n_init_qpos)
        if len(result) > 0:
            result = result
        else:
            result = None
        return status, result



    def step(self, ee_target: Pose, qpos: np.ndarray, move_base: bool=False, base_pose=None): #, last_ee: Optional[Pose]=None, last_qpos: Optional[np.ndarray]=None):
        last_ee, last_qpos = self.last_ee, self.last_qpos

        # if self.ee_bounds is not None:
        #     out = base_pose.inv() * ee_target
        #     out.p = np.clip(out.p, self.ee_bounds[0], self.ee_bounds[1])
        #     ee_target = base_pose * out

        self.block.set_pose(ee_target)
        self.show_axis(ee_target)


        if qpos is not None:
            self.qpos = qpos

        if ee_target is not None:
            # print('base pose', base_pose, move_base, qpos[:3])
            # if base_pose is not None:
            #    ee_target = cast(Pose, base_pose.inv() * ee_target)
            # print(base_pose, self.base_inv, self.qpos)
            # if not move_base:
            #     self.ee_target = base_pose.inv() * ee_target
            #     self.qpos[:3] = 0
            self.ee_target = cast(Pose, self.base_inv * ee_target)
            # print(self.ee_target)

            # # 1. try computing ik and see if we can fix the base with the current qpos
            # self.ik_status, self.ik_qpos = self.IK(self.ee_target, self.qpos, fix_base=True, n_init_qpos=10)

            # if self.ik_status != "Success":
            #     # 2. compute ik trying by sampling bases
            #     if last_ee is not None and last_qpos is not None:
            #         qpos = last_qpos

            #     if self.ik_status != "Success" and self.qpos is not qpos:
            #         self.ik_status, self.ik_qpos = self.IK(self.ee_target, qpos, fix_base=True, n_init_qpos=20)

            #     # 3. compute ik trying by sampling bases
            #     if move_base and self.ik_qpos is None:
            #         self.ik_status, self.ik_qpos = self.IK(self.ee_target, qpos, fix_base=False)
            self.ik_status, self.ik_qpos = self.IK(self.ee_target, self.qpos, fix_base=True, n_init_qpos=20)


            if self.ik_status == "Success":
                if move_base:
                    sol = self.ik_qpos[0] # type: ignore
                    self.base_axis.set_scale([0.1] * 3) # type: ignore
                    self.base_axis.set_position(np.append(sol[:2], 0))
                    self.base_axis.set_rotation(transforms3d.euler.euler2quat(0, 0, sol[2]))

                self.block_material.base_color = [0, 1, 0, 1]
            else:
                self.block_material.base_color = [1, 0, 0, 1]

        if self.ik_status == "Success":
            assert self.ik_qpos is not None
            ik_qpos = self.ik_qpos[0]
            self.fake_articulation.set_qpos(ik_qpos)
            arm_target = ik_qpos[self.arm_indices]
            arm_cur = self.qpos[self.arm_indices]

            #if self.path_cache is None or not self.path_cache.update(arm_cur, arm_target):

            if np.linalg.norm(arm_cur - arm_target) > 0.3 and False:
                # Require path simplification to be done before planning
                self.planner.robot.set_qpos(self.qpos[len(self.base_indices):], True)
                status, path = self.planner.planner.plan(
                    arm_cur, [arm_target], range=0.1, verbose=False, time=1.,
                )

                if status != "Exact solution":
                    # print("RRT Failed. %s" % status)
                    self.path_cache = None
                else:
                    # print("RRT succeed")
                    self.path_cache = PathCache(path[1:], self.planner)
            
            if self.path_cache is not None:
                arm_target = self.path_cache.path[0]
            self.last_ee, self.last_qpos = self.ee_target, ik_qpos

            return ik_qpos, arm_target
        else:
            return None, None

    

    def show_axis(self, pose: Optional[Pose]):
        if pose is None:
            for c in self.axis.children:
                c.transparency = 1 # type: ignore
        else:
            for c in self.axis.children:
                c.transparency = 0 # type: ignore
            self.axis.set_scale([0.1] * 3) # type: ignore
            self.axis.set_position(pose.p)
            self.axis.set_rotation(pose.q)
    
    def setup_sapien(self):
        sim = self.sim
        builder = sim._scene.create_actor_builder()
        self.block_material = sim._renderer.create_material()
        self.block_material.base_color = [1., 0., 0., 1.]
        builder.add_box_visual(Pose((0, 0, 0.)), (0.02, 0.02, 0.02), self.block_material)
        self.block = builder.build_kinematic()

        self.axis = _create_coordinate_axes(sim.viewer)



        self.base_axis = _create_coordinate_axes(sim.viewer)
        for c in self.base_axis.children:
            c.transparency = 0.5 # type: ignore
        self.base_axis.set_scale([0.1] * 3) # type: ignore