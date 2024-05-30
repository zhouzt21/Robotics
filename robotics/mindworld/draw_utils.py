import os
import transforms3d
import numpy as np
from mplib import Planner
from typing import Optional, cast
from robotics.sim import Simulator
from robotics import Pose
from sapien.utils.viewer import Viewer



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




def create_coordinate_axes(sim: Simulator, scale: float=0.1):
    base_axis = _create_coordinate_axes(sim.viewer)
    for c in base_axis.children:
        c.transparency = 0.5 # type: ignore
    base_axis.set_scale([scale] * 3) # type: ignore
    return base_axis

def set_coordinate_axes(base_axis, pose=None, scale: float=0.1):
    if pose is not None:
        for c in base_axis.children:
            c.transparency = 0 # type: ignore
        base_axis.set_scale([scale] * 3) # type: ignore
        base_axis.set_position(pose.p)
        base_axis.set_rotation(pose.q)
    else:
        for c in base_axis.children:
            c.transparency = 1 # type: ignore
    return base_axis