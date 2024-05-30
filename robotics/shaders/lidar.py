import sapien
from sapien import Entity, Pose
from sapien.render import RenderCameraComponent
from sapien.asset import create_dome_envmap
import numpy as np
import matplotlib.pyplot as plt
import transforms3d
import trimesh

def main():
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    # Build scene
    scene_config = sapien.SceneConfig()
    scene = engine.create_scene(scene_config)
    scene.set_environment_map(create_dome_envmap())
    scene.set_ambient_light([0.1, 0.1, 0.1])
    scene.add_point_light([0, 0, 0.5], [1, 1, 1])
    scene.add_directional_light([0, 0, -1], [1, 1, 1])
    ground_material = renderer.create_material()
    ground_material.base_color = np.array([202, 164, 114, 256]) / 256
    ground_material.specular = 0.5
    scene.add_ground(0, render_material=ground_material, render_half_size=[1.0, 10.0])
    scene.set_timestep(1 / 240)
    builder = scene.create_actor_builder()
    material = renderer.create_material()
    material.base_color = [0.2, 0.2, 0.8, 1.0]
    material.roughness = 0.5
    material.metallic = 0.0
    builder.add_sphere_visual(radius=0.06, material=material)
    builder.add_sphere_collision(radius=0.06)
    sphere1 = builder.build()
    sphere1.set_pose(Pose(p=[-0.05, 0.05, 0.06]))
    builder = scene.create_actor_builder()
    material = renderer.create_material()
    material.ior = 1.2
    material.transmission = 1.0
    material.base_color = [1.0, 1.0, 1.0, 1.0]
    material.roughness = 0.4
    material.metallic = 0.0
    builder.add_sphere_visual(radius=0.07, material=material)
    builder.add_sphere_collision(radius=0.07)
    sphere2 = builder.build()
    sphere2.set_pose(Pose(p=[0.05, -0.05, 0.07]))
    builder = scene.create_actor_builder()
    material = renderer.create_material()
    material.base_color = [0.8, 0.7, 0.1, 1.0]
    material.roughness = 0.1
    material.metallic = 1.0
    builder.add_capsule_visual(radius=0.02, half_length=0.1, material=material)
    builder.add_capsule_collision(radius=0.02, half_length=0.1)
    cap = builder.build()
    cap.set_pose(
        Pose(p=[0.15, -0.01, 0.01], q=transforms3d.euler.euler2quat(0, 0, -0.7))
    )
    builder = scene.create_actor_builder()
    material = renderer.create_material()
    material.base_color = [0.8, 0.2, 0.2, 1.0]
    material.roughness = 0.1
    material.metallic = 1.0
    builder.add_box_visual(half_size=[0.09, 0.09, 0.09], material=material)
    builder.add_box_collision(half_size=[0.09, 0.09, 0.09])
    box = builder.build()
    box.set_pose(Pose(p=[0.05, 0.17, 0.09]))

    # Camera mount
    camera_mount = Entity()
    camera_mount.set_pose(Pose([0.38052, 0.0489752, 0.1], [0.0083046, 0.0149983, 0.000115037, -0.999853]))
    scene.add_entity(camera_mount)

    # Normal RGB camera
    sapien.render.set_camera_shader_dir("rt")
    sapien.render.set_ray_tracing_samples_per_pixel(32)
    sapien.render.set_ray_tracing_path_depth(8)
    sapien.render.set_ray_tracing_denoiser("oidn")
    rgb_camera = RenderCameraComponent(512, 512)
    camera_mount.add_component(rgb_camera)

    # Lidar depth
    sapien.render.set_camera_shader_dir("lidar")
    lidar_camera = RenderCameraComponent(64, 64) # Divide 360 degree into 64*64 rays, how rows and cols are distributed is not important
    camera_mount.add_component(lidar_camera)
    sapien.render.set_camera_shader_dir("rt")
    scene.update_render()

    lidar_camera.take_picture()

    scene.update_render()
    rgb_camera.take_picture()

    rgb = rgb_camera.get_picture("Color")[..., :3]
    position1 = rgb_camera.get_picture("Position") # get_picture("Position") returns (x, y, z, depth) in camera frame
    points1 = position1[..., :3].reshape(-1, 3)

    position2 = lidar_camera.get_picture("Position")
    # Missed ray will have depth 0, so we need to remove them
    points2 = position2[..., :3].reshape(-1, 3)[position2[..., 3].reshape(-1) > 0]

    # Plot RGB picture of the scene
    plt.imshow(rgb)
    plt.show()

    # Plot point cloud
    cloud1 = trimesh.points.PointCloud(points1, colors=[255, 0, 0]) # Red
    cloud2 = trimesh.points.PointCloud(points2, colors=[0, 255, 0]) # Green
    trimesh_scene = trimesh.Scene()
    trimesh_scene.add_geometry(cloud1)
    trimesh_scene.add_geometry(cloud2)
    trimesh_scene.show()

main()
