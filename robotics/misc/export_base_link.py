import trimesh
import open3d as o3d
import numpy as np

mesh = trimesh.load('../assets/turtlebot4/body_visual.dae')
print(mesh.bounds)
assert  isinstance(mesh, trimesh.Scene), type(mesh)
pcd = []
for m in mesh.geometry:
    m = mesh.geometry[m]
    assert isinstance(m, trimesh.Trimesh), type(m)
    pcd.append(m.sample(100000))

pcd = np.concatenate(pcd) / 100.

np.save('turtlebot4_body.npy', pcd)
print(pcd.min(axis=0), pcd.max(axis=0))

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
o3d_point = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
o3d.visualization.draw_geometries([o3d_point, axis])