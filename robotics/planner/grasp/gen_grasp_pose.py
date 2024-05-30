# adopted from https://github.com/Jiayuan-Gu/ManiSkill2-solution/blob/main/grasp_pose/gen_ycb.py
import sapien
import pathlib
import sapien.render
import numpy as np
import open3d as o3d
import tqdm
from robotics import Pose
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation


def norm_vec(x, eps=1e-6):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.where(norm < eps, 0, x / np.maximum(norm, eps))


def rotate_transform(R):
    out = np.zeros(R.shape[:-2] + (4, 4))
    out[..., :3, :3] = R
    out[..., 3, 3] = 1
    return out




def compute_antipodal_contact_points(points, normals, max_width, score_thresh):
    dist = cdist(points, points)

    # Distance between two contact points should be larger than gripper width.
    mask1 = dist <= max_width

    # [n, n, 3], direction between two surface points
    direction = norm_vec(points[None, :] - points[:, None])
    # [n, n]
    cos_angle = np.squeeze(direction @ normals[..., None], -1)

    # Heuristic from S4G
    score = np.abs(cos_angle * cos_angle.T)
    mask2 = score >= score_thresh
    # print(score[0:5, 0:5])

    row, col = np.nonzero(np.triu(np.logical_and(mask1, mask2), k=1))
    return row, col, score[row, col]


def initialize_grasp_poses(points, row, col):
    # Assume the grasp frame is approaching (x), closing (y), ortho (z).
    # The origin is the center of two contact points.
    # Please convert to the gripper you use.

    # The closing vector is segment between two contact points
    displacement = points[col] - points[row]
    closing_vec = norm_vec(displacement)  # [m, 3]
    # Approaching and orthogonal vectors should be searched later.
    U, _, _ = np.linalg.svd(closing_vec[..., None])  # [m, 3, 3]
    approaching_vec = U[..., 1]  # [m, 3]
    assert np.all(np.einsum("nd,nd->n", approaching_vec, closing_vec) <= 1e-6)
    center = (points[col] + points[row]) * 0.5

    grasp_frames = np.tile(np.eye(4), [len(row), 1, 1])
    grasp_frames[:, 0:3, 0] = closing_vec
    grasp_frames[:, 0:3, 1] = approaching_vec
    grasp_frames[:, 0:3, 2] = np.cross(closing_vec, approaching_vec)
    grasp_frames[:, 0:3, 3] = center
    return grasp_frames, np.linalg.norm(displacement, axis=-1) * 0.5


def augment_grasp_poses(grasp_poses, angles):
    Rs = Rotation.from_euler("x", angles).as_matrix()  # [A, 3, 3]
    Ts = rotate_transform(Rs)  # [A, 4, 4]
    out = np.einsum("nij,mjk->nmik", grasp_poses, Ts)
    return out


from robotics.sim.robot.urdf import URDFTool
def extract_gripper(urdf_tool: URDFTool, ee_name: str):
    import os
    joint = None
    for i in urdf_tool.all_joints.values():
        if i.child == ee_name:
            joint = i.name
            break
    assert joint is not None, f"Cannot find the joint connecting to the end effector `{ee_name}`"
    urdf_tool.remove(joint, dtype='joint')
    urdf_tool = urdf_tool.prune_from(ee_name)

    path = 'gripper.urdf' # NOTE: currently it must has './' or '/' in the filename for mplib
    urdf_tool.export(path, absolute=True)
    return path, urdf_tool


from robotics.sim import Simulator, SimulatorConfig, CameraConfig
from robotics.mindworld.draw_utils import create_coordinate_axes, set_coordinate_axes
import xml.etree.ElementTree as ET


def get_gripper_urdf(urdf_path, srdf_path, ee_name: str):
    #assert robot is not None
    urdf_path = urdf_path
    urdf_tool = URDFTool.from_path(urdf_path)
    urdf_path, urdf_tool = extract_gripper(urdf_tool, ee_name)

    #srdf_path = robot.get_srdf_path()
    srdf_tree = ET.parse(srdf_path).getroot()
    for i in srdf_tree.findall('disable_collisions'):
        if i.attrib['link1'] not in urdf_tool.all_links or i.attrib['link2'] not in urdf_tool.all_links:
            srdf_tree.remove(i)

    srdf_path = urdf_path.replace('.urdf', '.srdf')
    srdf_tree = ET.ElementTree(srdf_tree)
    srdf_tree.write(srdf_path)
    return urdf_path, srdf_path


    
def get_gripper_in_sim(sim: Simulator, remove_articulation: bool=False, n_qpos: int=10, dy=0.06, dx=0.005):
    robot = sim.robot
    assert robot is not None
    # urdf_path = robot.urdf_path
    # urdf_tool = URDFTool.from_path(urdf_path)
    # urdf_path, urdf_tool = extract_gripper(urdf_tool, robot.ee_name)

    # srdf_path = robot.get_srdf_path()
    # srdf_tree = ET.parse(srdf_path).getroot()
    # for i in srdf_tree.findall('disable_collisions'):
    #     if i.attrib['link1'] not in urdf_tool.all_links or i.attrib['link2'] not in urdf_tool.all_links:
    #         srdf_tree.remove(i)

    # srdf_path = urdf_path.replace('.urdf', '.srdf')
    # srdf_tree = ET.ElementTree(srdf_tree)
    # srdf_tree.write(srdf_path)
    urdf_path, srdf_path = get_gripper_urdf(robot.urdf_path, robot.get_srdf_path(), robot.ee_name)


    loader = sim._scene.create_urdf_loader()
    loader.fix_root_link = True
    gripper = loader.load(urdf_path, srdf_path)
    gripper.set_pose(Pose([0., 0., 2.]))
    dof = gripper.dof

    for i in gripper.get_links():
        for j in i.entity.components:
            if isinstance(j, sapien.render.RenderBodyComponent):
                j.visibility = 0.3

    from blazar.utils.utils import timeit_context
    # ROBOT specific code for generating grasp poses with different width
    action = np.zeros((sim.action_space.shape[0],))
    with timeit_context('qpos'):
        # enumerate possible qpos
        qpos = []
        for i in np.linspace(-1., 1, n_qpos):
            action[-1] = i
            for j in range(10):
                sim.step(action)
            assert hasattr(robot, 'gripper')
            qpos.append(robot.articulation.get_qpos()[-dof:])

    # gripper_right1
    axis = create_coordinate_axes(sim)
    init_pose = gripper.get_pose()

    #init_pose.p[1] += 0.05
    p = init_pose.p
    p[0] += dx#0.005
    p[1] += dy#0.05
    p[2] -= 0.01
    init_pose.p = p
    set_coordinate_axes(axis, init_pose, scale=0.05)

    def find(name):
        for i in gripper.get_links():
            if i.name == name:
                return i
        raise ValueError(f'Cannot find link {name}')

    right = find('gripper_right1')
    right_contact = right.get_pose().inv() * init_pose
    left = find('gripper_left1')
    p[0] -= dx * 2
    init_pose.p = p
    left_contact = left.get_pose().inv() * init_pose

    tips = []

    base_inv = gripper.get_pose().inv()
    for i in qpos:
        gripper.set_qpos(i)
        ll = base_inv * left.get_pose() * left_contact
        rr = base_inv * right.get_pose() * right_contact
        tips.append((ll, rr))
    ll, rr = tips[0]
    root_to_tcp = Pose((ll.p + rr.p)/2, [1., 0., 0., 0.]).inv().to_transformation_matrix()

    if remove_articulation:
        sim._scene.remove_articulation(robot.articulation)
    return gripper, qpos, root_to_tcp


def main():
    from robotics.sim.robot.mycobot280pi import MyCobot280Arm
    from robotics.sim.environs.ycb_clutter import YCBClutter, YCBConfig
    # from xml import ElementTrees as ET

    robot = MyCobot280Arm(60, arm_controller='posvel')
    sim = Simulator(SimulatorConfig(contact_offset=1e-9, viewer_camera=CameraConfig(look_at=(0., 0., 0.5), p=(0.5, 0.5, 1.))), robot, {})
    sim.reset()

    gripper, qpos, root_to_tcp = get_gripper_in_sim(sim, remove_articulation=True)


    YCB_DIR = pathlib.Path('/root/RealRobot/realbot/assets/data/mani_skill2_ycb')
    model_name = '072-a_toy_airplane'
    # model_name = '048_hammer'
    mesh_path = str(YCB_DIR / f"models/{model_name}/collision.obj")
    textured_mesh_path = str(YCB_DIR / f"models/{model_name}/textured.obj")
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    mesh.compute_vertex_normals()
    # Open3d interpolates normals for sampled points.
    pcd = mesh.sample_points_uniformly(number_of_points=2048)
    points, normals = np.array(pcd.points), np.array(pcd.normals)


    row, col, score = compute_antipodal_contact_points(
        points, normals, 0.08 * 0.95, 0.97 # TODO: change max_width
    )
    print("#contact points", len(row))

    
    # grasp pose generation
    n_angles = 20
    grasp_poses, closing_dists = initialize_grasp_poses(points, row, col)
    angles = np.linspace(0, 2 * np.pi, n_angles)
    grasp_poses = augment_grasp_poses(grasp_poses, angles)
    grasp_poses = np.reshape(grasp_poses, [-1, 4, 4])
    closing_dists = np.repeat(closing_dists, n_angles, axis=0)


    # for visualization only
    actor_builder = sim._scene.create_actor_builder()
    actor_builder.add_visual_from_file(textured_mesh_path)


    # from realbot.utils.convexify import convexify
    # convex_path = textured_mesh_path + '.convex.obj'
    # mesh = convexify(textured_mesh_path, convex_path, method='vhacd', skip_if_exists=True)
    convex_path = mesh_path
    actor_builder.add_multiple_convex_collisions_from_file(convex_path, decomposition='coacd')
    actor = actor_builder.build_static()
    actor_pose = Pose([0., 0., 0.4])
    actor.set_pose(actor_pose)
    actor.name = 'ycb'


    pbar = tqdm.tqdm(total=len(grasp_poses))
    answer = []
    for grasp_pose in grasp_poses:
        root_pose = Pose(grasp_pose @ root_to_tcp)
        pbar.update()

        gripper.set_qpos(qpos[-1])
        gripper.set_pose(actor_pose * root_pose)
        actor.set_pose(actor_pose)
        sim._scene.step()
        after_qpos = gripper.get_qpos()
        if np.linalg.norm(after_qpos - qpos[-1]) > 1e-2: # type: ignore
            continue
        if len(sim._scene.get_contacts()) > 0:
            continue

        answer.append((root_pose, qpos[-1]))
        # tcp = gripper.get_pose() * Pose(root_to_tcp).inv()
        # set_coordinate_axes(axis, tcp, scale=0.05)
        for i in range(100):
            sim.render()

    print("#" * 100)
    print(len(answer))



    
if __name__ == "__main__":
    main()