#!/usr/bin/env python3
"""在 SAPIEN Viewer 中可视化 002_bowl/base3 和接触点 1、3。

默认读取：
  external/robotwin_local/assets/objects/002_bowl/model_data3.json
  external/robotwin_local/assets/objects/002_bowl/visual/base3.glb

运行（需要在已安装 SAPIEN 的 conda 环境中，并且有图形界面）：
  python tools/visualize_bowl_contact_points.py

接触点的局部平移会按模型 scale 缩放，这与 Actor.get_point() 完全一致。
可选 --contact-x/--contact-y 用于预览 stack_bowls_three.py 中的覆盖值；
不传时显示 model_data3.json 的原始接触点。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sapien.core as sapien
from sapien.utils.viewer import Viewer
import transforms3d.quaternions as t3q


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "external" / "robotwin_local" / "assets" / "objects" / "002_bowl"
MODEL_JSON = MODEL_DIR / "model_data3.json"
MODEL_GLB = MODEL_DIR / "visual" / "base3.glb"


def pose_matrix(pose: sapien.Pose) -> np.ndarray:
    return pose.to_transformation_matrix()


def add_marker_sphere(scene, position, color, radius=0.008):
    builder = scene.create_actor_builder()
    material = sapien.render.RenderMaterial()
    material.base_color = [*color, 1.0]
    # ActorBuilder 的球体可视化接口（与 RoboTwin 自带 create_sphere 一致）。
    builder.add_sphere_visual(radius=radius, material=material)
    actor = builder.build_static()
    actor.set_pose(sapien.Pose(position, [1, 0, 0, 0]))
    return actor


def add_axis_rod(scene, origin, direction, color, length=0.045, width=0.0025):
    """Add a colored rectangular rod whose local +x axis follows direction."""
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)

    # Build an orthonormal frame with local x = direction.
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, direction)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    y_axis = np.cross(helper, direction)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(direction, y_axis)
    rotation = np.column_stack([direction, y_axis, z_axis])

    material = sapien.render.RenderMaterial()
    material.base_color = [*color, 1.0]
    builder = scene.create_actor_builder()
    builder.add_box_visual(half_size=[length / 2, width, width], material=material)
    actor = builder.build_static()
    center = np.asarray(origin) + direction * (length / 2)
    actor.set_pose(sapien.Pose(center, t3q.mat2quat(rotation)))
    return actor


def add_contact_frame(scene, bowl_pose, local_contact, label_color):
    local = np.asarray(local_contact, dtype=float).copy()
    # Actor.get_point() scales only the local translation, not its rotation.
    local[:3, 3] *= np.asarray([0.05, 0.05, 0.05], dtype=float)
    world = pose_matrix(bowl_pose) @ local
    origin = world[:3, 3]
    rotation = world[:3, :3]

    add_marker_sphere(scene, origin, label_color)
    add_axis_rod(scene, origin, rotation[:, 0], [1.0, 0.1, 0.1])  # x: red
    add_axis_rod(scene, origin, rotation[:, 1], [0.1, 1.0, 0.1])  # y: green
    add_axis_rod(scene, origin, rotation[:, 2], [0.1, 0.3, 1.0])  # z: blue
    return origin, rotation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-x", type=float, default=None,
                        help="override local contact x before model scaling")
    parser.add_argument("--contact-y", type=float, default=None,
                        help="override local contact y before model scaling")
    parser.add_argument("--z", type=float, default=0.0,
                        help="bowl world z translation, default 0")
    args = parser.parse_args()

    if not MODEL_JSON.is_file() or not MODEL_GLB.is_file():
        raise SystemExit(f"找不到碗模型或 model_data3.json：{MODEL_DIR}")

    data = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
    scale = np.asarray(data["scale"], dtype=float)
    contacts = [np.asarray(p, dtype=float) for p in data["contact_points_pose"]]
    if args.contact_x is not None:
        for contact in contacts:
            contact[0, 3] = args.contact_x
    if args.contact_y is not None:
        for contact in contacts:
            contact[1, 3] = args.contact_y

    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)
    scene = engine.create_scene()
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, -1, -1], [1, 1, 1], shadow=True)
    scene.add_ground(altitude=-0.03)

    bowl_pose = sapien.Pose([0.0, 0.0, args.z], [0.5, 0.5, 0.5, 0.5])
    bowl_builder = scene.create_actor_builder()
    bowl_builder.set_physx_body_type("static")
    bowl_builder.add_visual_from_file(filename=str(MODEL_GLB), scale=scale.tolist())
    bowl = bowl_builder.build_static(name="002_bowl_base3")
    bowl.set_pose(bowl_pose)

    # Point 1 = right-side contact, point 3 = left-side contact in model_data3.
    colors = {1: [1.0, 0.75, 0.0], 3: [0.8, 0.1, 1.0]}
    for point_id in (1, 3):
        origin, rotation = add_contact_frame(scene, bowl_pose, contacts[point_id], colors[point_id])
        print(f"contact point {point_id}:")
        print("  local translation (before scale):", contacts[point_id][:3, 3].tolist())
        print("  local translation (after scale): ", (contacts[point_id][:3, 3] * scale).tolist())
        print("  world position:                   ", origin.tolist())
        print("  world rotation:\n", np.array2string(rotation, precision=5, suppress_small=True))

    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=0.28, y=-0.32, z=0.24)
    viewer.set_camera_rpy(r=0.0, p=-0.65, y=0.72)
    print("Viewer 已启动：红/绿/蓝分别是接触坐标系 x/y/z 轴，黄色=点1，紫色=点3。关闭窗口退出。")
    while not viewer.closed:
        scene.step()
        scene.update_render()
        viewer.render()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
