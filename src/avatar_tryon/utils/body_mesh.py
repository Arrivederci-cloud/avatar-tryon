"""参数化人体代理网格。

用途：在没有 LHM/MHR 权重、也没有 GPU 的环境里，给整条流水线一个
真实比例的几何锚点，使「单图 -> 三视图 -> UV 纹理 -> 试穿」可以完整跑通。

它不是重建结果，只是一个体型先验：
    躯干 / 头 / 颈 / 双臂 / 双腿 按身高与三围生成广义圆柱，
    每一段落在自己的 UV 矩形里（分块图集），缝线处复制顶点，避免 UV 环绕。

正式部署时这个模块被 LHM/MHR 的输出替换，下游代码完全不变——
这正是「几何来自参数化先验」这条设计原则的落地形态。
"""

from __future__ import annotations

import math
from pathlib import Path

from .mesh import write_obj

# 每个部件在 UV 图集里的矩形：(u0, u1, v0, v1)
ATLAS: dict[str, tuple[float, float, float, float]] = {
    "torso": (0.02, 0.36, 0.50, 0.98),
    "head": (0.38, 0.52, 0.78, 0.99),
    "neck": (0.38, 0.50, 0.62, 0.74),
    "arm_l": (0.54, 0.72, 0.52, 0.96),
    "arm_r": (0.74, 0.92, 0.52, 0.96),
    "leg_l": (0.02, 0.24, 0.02, 0.40),
    "leg_r": (0.26, 0.48, 0.02, 0.40),
}

SEGMENTS = {"torso": 48, "head": 28, "neck": 20, "limb": 24}


def _refine(rings, times: int):
    """在相邻环之间插入中点，增加纵向细分（解剖学关键环位置不变）。"""
    for _ in range(times):
        out = [rings[0]]
        for a, b in zip(rings, rings[1:], strict=False):
            out.append(tuple((x + y) / 2.0 for x, y in zip(a, b, strict=False)))
            out.append(b)
        rings = out
    return rings


def girth_to_axes(girth_cm: float, ratio: float) -> tuple[float, float]:
    """围度 -> 椭圆半轴 (ax, az)，ratio = ax/az。

    用拉马努金近似周长公式反解，二分 60 次足够收敛到微米级。
    """
    def circ(az: float) -> float:
        ax = ratio * az
        return math.pi * (3.0 * (ax + az) - math.sqrt((3.0 * ax + az) * (ax + 3.0 * az)))

    lo, hi = 0.5, max(girth_cm, 2.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if circ(mid) < girth_cm:
            lo = mid
        else:
            hi = mid
    return ratio * (lo + hi) / 2.0, (lo + hi) / 2.0


class _Builder:
    def __init__(self) -> None:
        self.verts: list[tuple[float, float, float]] = []
        self.uvs: list[tuple[float, float]] = []
        self.faces: list[tuple[list[int], list[int]]] = []

    def ring(self, pts, u0: float, u1: float, v: float) -> int:
        """加入一圈顶点（末点与首点重合，uv 走到 u1 不回头），返回 1-based 起始索引。"""
        base = len(self.verts) + 1
        n = len(pts)
        for k, p in enumerate(pts):
            self.verts.append(p)
            self.uvs.append((u0 + (u1 - u0) * k / (n - 1), v))
        return base

    def skin(self, a0: int, b0: int, n: int) -> None:
        """连接两圈等长顶点。"""
        for s in range(n - 1):
            q = [a0 + s, a0 + s + 1, b0 + s + 1, b0 + s]
            self.faces.append(([q[0], q[2], q[1]], [q[0], q[2], q[1]]))
            self.faces.append(([q[0], q[3], q[2]], [q[0], q[3], q[2]]))

    def tube(self, rings, segments: int, rect, subdiv: int = 1) -> None:
        """rings: [(cx, y, cz, ax, az), ...] 自上而下或自下而上均可。"""
        u0, u1, v0, v1 = rect
        rings = _refine(rings, subdiv)
        starts = []
        for i, (cx, y, cz, ax, az) in enumerate(rings):
            v = v0 + (v1 - v0) * (i / (len(rings) - 1))
            pts = [
                (cx + ax * math.sin(t), y, cz + az * math.cos(t))
                for t in (2.0 * math.pi * s / segments for s in range(segments + 1))
            ]
            starts.append(self.ring(pts, u0, u1, v))
        for i in range(len(starts) - 1):
            self.skin(starts[i], starts[i + 1], segments + 1)


def build_body(
    height_cm: float,
    chest_cm: float,
    waist_cm: float,
    hip_cm: float,
    arm_out: float = 1.0,
):
    """生成人体代理网格，返回 (verts, uvs, faces)。单位厘米，脚底在 y=0。

    arm_out：手臂外展系数。1.0 是标准 A-pose 先验；垂手站立的真人照片里
    手臂贴着身体，实测需要 0.7~0.85 才能让网格轮廓与照片对齐——
    手臂位置不对时，投影烘焙会把手臂 texel 采样到背景上直接丢弃。
    """
    h = float(height_cm)
    chest_a, chest_z = girth_to_axes(chest_cm, 1.35)
    waist_a, waist_z = girth_to_axes(waist_cm, 1.30)
    hip_a, hip_z = girth_to_axes(hip_cm, 1.40)
    shoulder_half = max(0.118 * h, chest_a * 1.10)
    neck_r = 0.028 * h
    thigh_ax, thigh_az = hip_a * 0.52, hip_z * 0.56

    b = _Builder()

    # 躯干：裆部 -> 臀 -> 腰 -> 胸 -> 腋 -> 斜方肌过渡 -> 颈根
    # 肩部不做平直的「搁板」：宽度渐收，斜方肌处明显变窄
    b.tube(
        [
            (0.0, 0.470 * h, 0.0, hip_a * 0.92, hip_z * 0.92),
            (0.0, 0.530 * h, 0.0, hip_a, hip_z),
            (0.0, 0.620 * h, 0.0, waist_a, waist_z),
            (0.0, 0.720 * h, 0.0, chest_a, chest_z),
            (0.0, 0.780 * h, 0.0, chest_a * 0.98, chest_z * 0.95),
            (0.0, 0.805 * h, 0.0, chest_a * 1.05, chest_z * 0.92),
            (0.0, 0.825 * h, 0.0, shoulder_half * 0.97, chest_z * 0.85),
            (0.0, 0.840 * h, 0.0, shoulder_half * 0.72, chest_z * 0.55),
            (0.0, 0.852 * h, 0.0, neck_r * 1.3, neck_r * 1.3),
            (0.0, 0.858 * h, 0.0, neck_r * 1.05, neck_r * 1.05),
        ],
        SEGMENTS["torso"],
        ATLAS["torso"],
        2,
    )

    # 颈
    b.tube(
        [
            (0.0, 0.845 * h, 0.0, neck_r * 1.08, neck_r * 1.08),
            (0.0, 0.885 * h, 0.0, neck_r, neck_r),
        ],
        SEGMENTS["neck"],
        ATLAS["neck"],
        1,
    )

    # 头：椭球，按纬度收窄，首末圈接近闭合
    head_cy, head_rx, head_ry, head_rz = 0.930 * h, 0.048 * h, 0.062 * h, 0.055 * h
    lats = 7
    head_rings = []
    for i in range(lats):
        t = math.pi * (i / (lats - 1))  # 0=顶, pi=底
        s = max(math.sin(t), 0.12)
        head_rings.append(
            (0.0, head_cy + head_ry * math.cos(t), 0.0, head_rx * s, head_rz * s)
        )
    b.tube(head_rings, SEGMENTS["head"], ATLAS["head"], 1)

    # 双腿：大腿 -> 膝 -> 小腿肚 -> 踝，保留小腿弧线，末端不收成锥尖
    for side, key in ((-1, "leg_l"), (1, "leg_r")):
        cx = side * hip_a * 0.50
        b.tube(
            [
                (cx, 0.505 * h, 0.0, thigh_ax, thigh_az),
                (cx, 0.430 * h, 0.0, thigh_ax * 0.95, thigh_az * 0.95),
                (cx, 0.340 * h, 0.0, 0.054 * h, 0.056 * h),
                (cx, 0.285 * h, 0.0, 0.048 * h, 0.050 * h),
                (cx, 0.180 * h, 0.0, 0.046 * h, 0.048 * h),
                (cx, 0.090 * h, 0.0, 0.033 * h, 0.035 * h),
                (cx, 0.020 * h, 0.0, 0.028 * h, 0.030 * h),
            ],
            SEGMENTS["limb"],
            ATLAS[key],
            1,
        )

    # 双臂：肩关节 -> 肘 -> 腕，外展量由 arm_out 控制
    for side, key in ((-1, "arm_l"), (1, "arm_r")):
        sx = side * shoulder_half * 0.92 * arm_out
        wx = side * shoulder_half * 1.15 * arm_out
        b.tube(
            [
                (sx, 0.820 * h, 0.0, 0.042 * h, 0.042 * h),
                (sx + (wx - sx) * 0.25, 0.730 * h, 0.0, 0.036 * h, 0.036 * h),
                (sx + (wx - sx) * 0.50, 0.625 * h, 0.0, 0.032 * h, 0.032 * h),
                (sx + (wx - sx) * 0.75, 0.530 * h, 0.0, 0.028 * h, 0.028 * h),
                (wx, 0.440 * h, 0.0, 0.022 * h, 0.022 * h),
            ],
            SEGMENTS["limb"],
            ATLAS[key],
            1,
        )

    return b.verts, b.uvs, b.faces


def write_body(
    path: Path,
    height_cm: float,
    chest_cm: float,
    waist_cm: float,
    hip_cm: float,
    arm_out: float = 1.0,
) -> Path:
    verts, uvs, faces = build_body(height_cm, chest_cm, waist_cm, hip_cm, arm_out)
    return write_obj(path, verts, uvs, faces)


__all__ = ["ATLAS", "build_body", "girth_to_axes", "write_body"]
