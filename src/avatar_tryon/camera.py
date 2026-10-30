"""固定机位约定。

单独成一个模块：三视图生成（pipeline）与软件渲染（models）都要用，
放在任一侧都会造成循环导入。

约定：
    front = yaw  0 度，相机在 +z，人物面向 +z
    side  = yaw 90 度，相机在 +x（人物右侧）
    back  = yaw 180 度，相机在 -z
三个机位不随机、不抖，保证三视图之间可以逐像素对齐。
"""

from __future__ import annotations

import math

VIEW_YAW = {"front": 0.0, "side": 90.0, "back": 180.0}


def camera_dirs() -> dict[str, tuple[float, float, float]]:
    """各机位相机所在的单位方向（从人体指向相机）。"""
    dirs = {}
    for name, yaw in VIEW_YAW.items():
        r = math.radians(yaw)
        dirs[name] = (math.sin(r), 0.0, math.cos(r))
    return dirs


__all__ = ["VIEW_YAW", "camera_dirs"]
