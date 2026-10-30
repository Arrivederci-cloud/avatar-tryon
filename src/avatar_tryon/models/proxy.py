"""参数化体型先验适配器（代理几何，不需要 GPU / 权重）。

定位要说清楚：它不给「这个人长什么样」，只给「这个身高的标准体型是什么比例」。
它的价值在于让 UV 链路、试衣链路在没有重建模型时就能全流程验证，
并且在 LHM/MHR 迟到时充当降级方案。

接入 LHM/MHR 后，同一份下游代码直接吃真实网格，接口不变。
"""

from __future__ import annotations

import math
from pathlib import Path

from ..utils.body_mesh import build_body, write_body
from ..utils.matte import person_matte
from ..utils.silhouette import BANDS, mesh_ranges, width_profile
from .base import HumanGeometry, ReconModel

# 身高 -> 三围的经验比例（亚洲成人平均），仅在缺少测量输入时使用
GIRTH_RATIOS = {"chest": 0.525, "waist": 0.425, "hip": 0.560}

# 拟合搜索空间：围度缩放 × 手臂外展
GIRTH_SCALES = (0.90, 0.96, 1.00, 1.06, 1.12)
ARM_OUTS = (0.70, 0.80, 0.90, 1.00)


def _fit_to_silhouette(image: Path, girths: dict[str, float], h: float):
    """把参数化几何拟合到照片轮廓，返回 (girths, arm_out, notes)。

    为什么要拟合：手臂外展是代理几何最大的系统误差。标准 A-pose 先验的
    手臂末端到中线约 27cm，垂手站立的真人只有 ~20cm；不修正时投影烘焙
    会把手臂外沿 texel 采样到背景上（实测手臂区残留 ~50%）。

    方法：对 (围度缩放, 手臂外展) 做网格搜索，最小化网格逐带半宽与
    GrabCut 轮廓逐带半宽的绝对偏差（按身高定像素/厘米比例）。
    只依赖已算好的轮廓剖面，不需要姿态关键点，也不需要额外权重。

    对齐细节：
    * 尺度用「人物像素高 / 网格身高跨度」而不是最大带宽——后者会把
      手臂外展误差吸收进尺度里，让围度拟合失去参照。
    * 跳过顶部 15% 与底部 6%：顶部是头发（网格没有头发，宽度不可比），
      底部是鞋；这些区域的差异与体型无关，混进来会把围度越推越大。
    """
    notes: list[str] = []
    try:
        from PIL import Image
    except ImportError:
        return girths, 1.0, notes

    try:
        src = Image.open(image).convert("RGB")
    except OSError:
        return girths, 1.0, notes

    matte, box = person_matte(src)
    prof = width_profile(matte) if matte is not None else None
    if prof is None:
        return girths, 1.0, notes

    # 采样带：15%~94%，避开头发与鞋
    lo_band, hi_band = int(BANDS * 0.15), int(BANDS * 0.94)
    target: list[tuple[int, float]] = []
    for b in range(lo_band, hi_band):
        if prof.lo[b] is None:
            continue
        half = (prof.hi[b] - prof.lo[b]) / 2.0
        if half > 2.0:
            target.append((b, half))
    if len(target) < (hi_band - lo_band) // 2:  # 轮廓太碎，不可信
        return girths, 1.0, notes
    t_max = max(t[1] for t in target)

    def cost(chest: float, waist: float, hip: float, arm_out: float) -> float:
        verts, _uvs, _faces = build_body(h, chest, waist, hip, arm_out)
        ys = [v[1] for v in verts]
        ymin, ymax = min(ys), max(ys)
        span = (ymax - ymin) or 1.0
        # 网格身高跨度对应照片人物像素高（含头发，故分母用跨度而非 h）
        scale = (box[3] - box[1]) / span
        lo, hi = mesh_ranges(verts, (1.0, 0.0, 0.0), ymin, span, BANDS)
        err = 0.0
        for b, half in target:
            if lo[b] is None:
                continue
            m_half = (hi[b] - lo[b]) / 2.0 * scale
            err += abs(m_half - half)
        return err / max(len(target), 1) / t_max

    best = (math.inf, dict(girths), 1.0)
    for gs in GIRTH_SCALES:
        scaled = {k: v * gs for k, v in girths.items()}
        for ao in ARM_OUTS:
            c = cost(scaled["chest"], scaled["waist"], scaled["hip"], ao)
            if c < best[0]:
                best = (c, scaled, ao)

    notes.append(
        f"几何拟合：围度 ×{best[1]['chest'] / girths['chest']:.2f}、"
        f"手臂外展 ×{best[2]:.2f}（逐带轮廓平均偏差 {best[0]:.1%}）"
    )
    return best[1], best[2], notes


class ProxyBodyRecon(ReconModel):
    name = "proxy-body"
    license = "Apache-2.0（本仓库原创）"

    def __init__(self, weights_dir: Path, measurements: dict[str, float] | None = None) -> None:
        self.weights_dir = weights_dir
        self.measurements = dict(measurements or {})

    def reconstruct(self, image: Path, height_cm: float) -> HumanGeometry:
        h = float(height_cm)
        chest = self.measurements.get("chest") or h * GIRTH_RATIOS["chest"]
        waist = self.measurements.get("waist") or h * GIRTH_RATIOS["waist"]
        hip = self.measurements.get("hip") or h * GIRTH_RATIOS["hip"]
        notes = [
            "代理几何：按身高与经验围度比例生成，非重建结果",
            "围度无法从单图观测，尺寸仅用于尺码分档",
        ]

        girths = {"chest": chest, "waist": waist, "hip": hip}
        arm_out = 1.0
        if image is not None and Path(image).exists():
            girths, arm_out, fit_notes = _fit_to_silhouette(image, girths, h)
            notes.extend(fit_notes)

        mesh = write_body(
            image.parent / "geometry" / "body.obj",
            h, girths["chest"], girths["waist"], girths["hip"], arm_out,
        )
        return HumanGeometry(
            mesh_path=mesh,
            measurements={
                "height": round(h, 1),
                "chest": round(girths["chest"], 1),
                "waist": round(girths["waist"], 1),
                "hip": round(girths["hip"], 1),
                "arm_out": round(arm_out, 2),
            },
            joints_cm={
                "head": (0.0, 0.0, 0.93 * h),
                "shoulder": (0.0, 0.0, 0.83 * h),
                "hip": (0.0, 0.0, 0.53 * h),
                "knee": (0.0, 0.0, 0.285 * h),
                "ankle": (0.0, 0.0, 0.055 * h),
            },
            scale_cm_per_unit=1.0,
            notes=notes,
        )


__all__ = ["GIRTH_RATIOS", "ProxyBodyRecon"]
