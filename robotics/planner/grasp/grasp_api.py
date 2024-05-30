# find grasp pose for an object in sapien scene
import tqdm
import sapien
import open3d as o3d
from robotics.sim import Simulator, SimulatorConfig, CameraConfig
from robotics.utils.sapien_utils import get_actor_mesh, get_rigid_dynamic_component
from robotics.mindworld.draw_utils import create_coordinate_axes, set_coordinate_axes
from robotics import Pose
from contextlib import contextmanager
import xml.etree.ElementTree as ET
import numpy as np
from .gen_grasp_pose import get_gripper_in_sim, compute_antipodal_contact_points, initialize_grasp_poses, augment_grasp_poses


class GenGraspPose:
    def __init__(self, sim: Simulator) -> None:
        self.sim = sim
        self.gripper, self.qpos, self.root_to_tcp = get_gripper_in_sim(
            self.sim, remove_articulation=False
        )

        
    def propose_grasp_pose(self, points, normals, n_angles: int=20):
        row, col, score = compute_antipodal_contact_points(
            points, normals, 0.08 * 0.95, 0.97 # TODO: change max_width
        )
        print("#contact points", len(row))

        
        # grasp pose generation
        grasp_poses, closing_dists = initialize_grasp_poses(points, row, col)
        angles = np.linspace(0, 2 * np.pi, n_angles)
        grasp_poses = augment_grasp_poses(grasp_poses, angles)
        grasp_poses = np.reshape(grasp_poses, [-1, 4, 4])
        closing_dists = np.repeat(closing_dists, n_angles, axis=0)
        return grasp_poses, closing_dists

    @contextmanager
    def load_gripper(self, verbose: int=0):
        robot = self.sim.robot

        assert robot is not None
        # self.sim.remove_articulation(robot.articulation)
        # self.sim.add_articulation(self.gripper)


        yield self.gripper
        if verbose:
            self.sim._set_viewer_camera(self.sim.config.viewer_camera)

        # self.sim.remove_articulation(self.gripper)
        # self.sim.add_articulation(robot.articulation)
        self.gripper.set_pose(Pose([100, 0, 100], [1, 0, 0, 0]))

    def get_grasp_pose(self, actor: sapien.Entity, verbose: int=0, avoid_contact: bool=False, n_angles: int=20):
        with self.load_gripper(verbose) as gripper:
            p = actor.get_pose().p
            if verbose:
                self.sim._set_viewer_camera(
                    CameraConfig(
                        p=p + np.array([0., 0., 0.5]),
                        look_at=tuple(p),
                    )
                )



            mesh = get_actor_mesh(actor, transform=True)
            assert mesh is not None
            #grasp_posenormal = mesh.normals

            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

            o3d_mesh.compute_vertex_normals()
            pcd = o3d_mesh.sample_points_uniformly(number_of_points=2048)
            points, normals = np.array(pcd.points), np.array(pcd.normals)
            self.points = points

            grasp_poses, _ = self.propose_grasp_pose(points, normals, n_angles=n_angles)


            pbar = tqdm.tqdm(total=len(grasp_poses))
            answer = []

            rigid_component = get_rigid_dynamic_component(actor)
            actor_pose = actor.get_pose()
            assert rigid_component is not None

            qpos = self.qpos[-1]

            for i in gripper.get_links():
                i.disable_gravity = True

            for i in gripper.get_active_joints():
                i.set_drive_properties(0., 0.)

            for grasp_pose in grasp_poses:
                root_pose = Pose(grasp_pose @ self.root_to_tcp)
                pbar.update()

                gripper.set_qpos(qpos)
                gripper.set_pose(root_pose)
                # gripper.set_qvel(np.zeros_like(qpos))
                # gripper.set_root_angular_velocity(np.zeros(3, dtype=np.float32)) # type: ignore
                # gripper.set_root_velocity(np.zeros(3, dtype=np.float32)) # type: ignore

                actor.set_pose(actor_pose)
                # rigid_component.set_linear_velocity(np.zeros(3)) # type: ignore
                # rigid_component.set_angular_velocity(np.zeros(3)) # type: ignore
                # self.sim._scene.step()
                # after_qpos = gripper.get_qpos()
                # # print(after_qpos, qpos)

                # if np.linalg.norm(after_qpos - qpos) > 1e-2: # type: ignore
                #     continue
                # if avoid_contact and len(self.sim._scene.get_contacts()) > 0:
                #     continue
                if self.sim.check_no_collision(gripper, actor, avoid_contact=avoid_contact):
                    answer.append((root_pose, qpos))
                    if verbose:
                        # tcp = gripper.get_pose() * Pose(root_to_tcp).inv()
                        # set_coordinate_axes(axis, tcp, scale=0.05)
                        for i in range(verbose):
                            self.sim.render()
        return answer

        