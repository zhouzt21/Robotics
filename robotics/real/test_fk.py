import sapien


from sapien.utils import Viewer
import os
import numpy as np
from sapien.sensor import StereoDepthSensor, StereoDepthSensorConfig
import sapien.core as sapien
from sapien.core import Pose


def main():
    urdf = 'mycobot_pi_v2/xx.urdf'

    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)


    scene = engine.create_scene()  # Create an instance of simulation world (aka scene)
    scene.set_timestep(1 / 100.0)  # Set the simulation frequency
    scene.add_ground(altitude=0)  # Add a ground
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])


    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(urdf) # replace it with any urdf

    

    qpos = [-1.04, -1.141, -0.104, -0.336, -0.003, -0.345] + [0] * 6
    robot.set_qpos(qpos)

    pmodel = robot.create_pinocchio_model()
    assert np.allclose(qpos, robot.get_qpos())

    pmodel.compute_forward_kinematics(qpos)
    for i, link in enumerate(robot.get_links()[:9]):
        print(i, link.name, pmodel.get_link_pose(i), link.pose)


if __name__ == '__main__':
    main()