import numpy as np
from robotics import Pose
from robotics.sim.simulator import Simulator
from robotics import Pose

from typing import Optional, List
from numpy.typing import NDArray

from dataclasses import dataclass, field


@dataclass
class CADModel:
    texture_file: Optional[str]
    collision_file: Optional[str]
    density: float = 1000
    scale: float = 1.0
    bbox: Optional[List[List[float]]] = None # min, max
    name: Optional[str] = None
    category: Optional[str] = None

    def load_into_simulator(self, sim: Simulator, rescale: float = 1.):
        assert self.texture_file is not None
        builder = sim._scene.create_actor_builder()

        scale = (self.scale * rescale,) * 3
        builder.add_multiple_convex_collisions_from_file(
            filename=self.collision_file,
            scale=scale,
            material=None,
            density=self.density,
            decomposition='coacd'
        )

        builder.add_visual_from_file(filename=self.texture_file, scale=scale)
        actor = builder.build()
        if self.name is not None:
            actor.name = self.name
        return actor

    def load_trimesh(self):
        assert self.texture_file is not None
        from robotics.utils.convexify import as_mesh
        mesh = as_mesh(self.texture_file)
        if mesh is None:
            raise ValueError(f'Empty mesh {self.texture_file}')
        mesh.apply_scale(self.scale)
        #assert isinstance(mesh, trimesh.Trimesh)
        return mesh



@dataclass
class WorldState:
    robot_name: str
    robot_qpos: Optional[np.ndarray]
    object_list: List[CADModel] | None = None
    object_pose: List[Pose] = field(default_factory=list)
    
    # TODO: other articulated objects

    # TODO: not used now ..
    robot_qvel: Optional[np.ndarray] = None
    object_vel: Optional[List[np.ndarray]] = None

    updated: bool = False

    def add_object(self, model: CADModel):
        if self.object_list is None:
            self.object_list = []
        self.object_list.append(model)
        self.object_pose.append(Pose())
        self.updated = True
        return len(self.object_list) - 1

    def update_object_pose(self, idx: int, pose: Pose):
        self.object_pose[idx] = pose
        self.updated = True

    def set_qpos(self, qpos: NDArray):
        self.robot_qpos = qpos
        self.updated = True