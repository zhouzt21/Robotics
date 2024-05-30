from robotics.mindworld.mind import Mind
import tqdm
import pickle
import copy
import numpy as np
import time
import os
from robotics.ros import ROSNode
from robotics.ros.nav import slam, localization, nav_goal
from robotics.mindworld.plugin import RGBDVisualizer, Plugin
from typing import List, cast
from robotics import Pose

from vision.ycb_scene.gen_ycb_scene import load_ycb_models, CADModel
from std_msgs.msg import Float64MultiArray

from robotics.planner.grasp import GenGraspPose



class GTPoseListener(Plugin):
    """
    listen to the ground truth pose of objects through /gt_object_pose

    format:
    - model_id
    - scale
    - p
    - q
    """
    def __init__(self, object_list: List[CADModel]) -> None:
        super().__init__()
        self.object_list = object_list

    def setup(self, mind: Mind):
        node = mind.node
        self._actors = []

        def callback(msg: Float64MultiArray):
            data = np.array(msg.data, dtype=np.float64).reshape((-1, 9))

            if len(self._actors) == 0:
                for i in range(len(data)):
                    model_id = int(round(data[i, 0]))
                    cad = self.object_list[model_id]
                    cad.scale = data[i, 1]

                    cad = copy.deepcopy(cad)

                    with mind.acquire_state() as state:
                        idx = state.add_object(cad)
                    self._actors.append(idx)
            
            for i in range(len(data)):
                actor = self._actors[i]

                p = data[i, 2:5]
                q = data[i, 5:]

                with mind.acquire_state() as state:
                    state.update_object_pose(actor, Pose(p, q))


        node.create_subscription(Float64MultiArray, '/gt_object_pose', callback, 1)


def main():
    import os
    objects = load_ycb_models()
    plugin = RGBDVisualizer()
    plugin = GTPoseListener(objects)
    mind = Mind(plugins=[plugin], ROS_DOMAIN_ID=1, use_sim_time=True)

    # ps -ef|grep slam|awk -F' ' '{print $2}' |xargs kill -9
    os.system("pkill nav2 -9")
    os.system("pkill slam -9")
    ros_start = time.time()

    node = mind.node
    slam_server = slam.SLAMToolbox(node, verbose=True)
    navigator = nav_goal.GoalNavigator(node, verbose=False)

    while not mind.viewer.closed:
        mind.render(hz=30., render=False)
        if len(mind._actors) > 0:
            break

    gripper_api = GenGraspPose(mind.sim)
    actor = None
    with mind.acquire_state() as state:
        for idx, i in enumerate(state.object_list):
            if i.name == '073-a_lego_duplo':
                actor = mind._actors[idx]
    assert actor is not None, 'cannot find actor'


    use_cache = True
    import os
    if os.path.exists('grasp_poses') and use_cache:
        grasp_poses = pickle.load(open('grasp_poses', 'rb'))
    else:
        grasp_poses = gripper_api.get_grasp_pose(actor, 0, n_angles=5)
        if use_cache:
            pickle.dump(grasp_poses, open('grasp_poses', 'wb'))
        
    
    from robotics.planner.ik import IKSolver
    ik = IKSolver(mind.sim, mind.robot.ee_name)

    ik.setup_sapien()
    ik.add_scene_collision(mind.gen_pcd())

    robot = mind.robot
    qpos = None
    actor_pose = actor.get_pose()
    for i in robot.articulation.get_links():
        i.disable_gravity = True
    for i in robot.articulation.get_active_joints():
        i.set_drive_properties(stiffness=0, damping=0)


    def compute_ik(grasp_pose, gripper_qpos):
        qpos = robot.articulation.get_qpos()
        qpos[-len(gripper_qpos):] = gripper_qpos
        robot.articulation.set_qpos(qpos)
        ik.qmask[-len(gripper_qpos):] = True
        status, qpos = ik.ik_move_base(grasp_pose, (0.2, 0.3))
        pbar.update(1)

        
        if status == 'Success':
            actor.set_pose(actor_pose)
            robot.articulation.set_qpos(qpos)

            qq = qpos.copy(); qq[:2] = -8
            ik.fake_articulation.set_qpos(qq)
            if mind.sim.check_no_collision(robot.articulation, actor, avoid_contact=False):
                return qpos
            # for i in range(10):
            #     mind.render(hz=30., render=True)
        return None


    use_qpos_cache = True
    if os.path.exists('qpos') and use_qpos_cache:
        qpos = pickle.load(open('qpos', 'rb'))
    else:
        pbar = tqdm.tqdm(total=len(grasp_poses))
        best = 0
        best_qpos = 0
        for i, gripper_qpos in grasp_poses:
            qpos = compute_ik(i, gripper_qpos)
            if qpos is not None:
                dist = np.linalg.norm(qpos[:2] - actor_pose.p[:2])
                if dist > best:
                    best_qpos = qpos
                    best = dist
                if dist > 0.24:
                    break

        qpos = best_qpos
        if use_qpos_cache:
            with open('qpos', 'wb') as f:
                pickle.dump(qpos, f)

    if qpos is None:
        raise RuntimeError('cannot find a valid ik for grasp pose')


    if True:
        t = qpos[:2] - actor_pose.p[:2]
        target = actor_pose.p[:2] + t * 4
        time.sleep(max(0, 5. - (time.time() - ros_start))) # wait for ros to start
        future = navigator.move_to(*target, qpos[2], sync=False)
        while not future.done():
            mind.render(hz=30., render=True, step=True)

    robot.set_gripper(1.0)
    ik.fake_articulation.set_qpos(qpos)

    robot = mind.robot
    while not robot.move_to(qpos, 0.02, 0.02, p=np.array([0.4, 0., 0.4])):
        mind.render(hz=30., render=True, step=True)

    cur_qpos = robot.articulation.get_qpos()

    ik.planner.robot.set_qpos(cur_qpos[3:], True)
    ik.planner.planning_world.set_use_point_cloud(True) # NOTE: collision check bugs .. 

    status, path = ik.planner.planner.plan(
        cur_qpos[3:9], [qpos[3:9]], range=0.1, verbose=False, time=10.,
    )

    assert status == 'Exact solution', f'cannot find a path, status: {status}'
    print(path)

    cur = 0
    while True:
        # print(qpos[:3], robot.articulation.get_qpos()[:3])
        if cur < len(path):
            if np.linalg.norm(path[cur] - robot.articulation.get_qpos()[3:9]) < 0.01:
                cur = cur + 1
        if cur == len(path):
            break
        # robot.qaction_pub.publish(path[cur])
        robot.send_qaction(path[cur])
        mind.render(hz=30., render=True)

    robot.set_gripper(-1.)
    for i in range(3 * 30):
        mind.render(hz=30., render=True)

    robot.send_qaction(cur_qpos[3:9])
    while True:
        mind.render(hz=30., render=True)

if __name__ == '__main__':
    main()