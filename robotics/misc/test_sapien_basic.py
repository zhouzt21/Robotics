import numpy as np
import sapien as sapien
from sapien.utils import Viewer

def main():
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)


    scene = engine.create_scene()  # Create an instance of simulation world (aka scene)
    scene.set_timestep(1 / 100.0)  # Set the simulation frequency
    scene.add_ground(altitude=0)  # Add a ground
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_camera('mycamera', width=100, height=100, fovy=1., near=0.1, far=100)

    viewer = Viewer(renderer)  # Create a viewer (window)
    viewer.set_scene(scene)  # Bind the viewer and the scene
    viewer.set_camera_xyz(x=-4, y=0, z=2)
    viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)

    while True:  # Press key q to quit
        scene.step()  # Simulate the world
        scene.update_render()  # Update the world to the renderer
        viewer.render()
        print('123')


if __name__ == '__main__':
    main()
