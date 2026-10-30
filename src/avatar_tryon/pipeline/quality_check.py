"""输入质检与尺度标定。

单图最大的坑不是画质，是尺度歧义：同一张照片可以是高个子站得远，
也可以是矮个子站得近。因此这里的第一件事不是打分，是把像素换算成厘米。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..utils.image import image_size

MIN_SIDE = 512
MIN_OCCUPANCY = 0.45
MAX_OCCUPANCY = 0.98
PLAUSIBLE_SCALE = (0.03, 0.60)  # 厘米 / 像素


@dataclass(slots=True)
class QualityReport:
    passed: bool
    scale_cm_per_px: float
    person_height_px: float
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def check(
    image: Path,
    height_cm: float,
    occupancy: float = 0.85,
    person_height_px: float | None = None,
) -> QualityReport:
    """occupancy 为人物高度占画面高度的比例。

    真实实现应由分割模型给出；没有分割时用经验值估计，并在 warnings 里标注。
    """
    w, h = image_size(image)
    reasons: list[str] = []
    warnings: list[str] = []

    if min(w, h) < MIN_SIDE:
        reasons.append(f"分辨率过低：{w}x{h}，最短边需不小于 {MIN_SIDE}")

    if not (MIN_OCCUPANCY <= occupancy <= MAX_OCCUPANCY):
        reasons.append(f"人物占比 {occupancy:.0%} 不在可用区间 {MIN_OCCUPANCY:.0%}-{MAX_OCCUPANCY:.0%}")

    person_px = person_height_px if person_height_px else h * occupancy
    scale = height_cm / person_px if person_px else 0.0

    lo, hi = PLAUSIBLE_SCALE
    if not (lo <= scale <= hi):
        reasons.append(f"尺度换算异常：{scale:.3f} cm/px 超出合理区间 {lo}-{hi}，请检查身高填写")

    if height_cm <= 0:
        reasons.append("身高未填写或非法，单图无法确定绝对尺寸")

    if person_height_px is None:
        warnings.append("未提供人物像素高度，按经验占比估计，围度结果仅供参考")

    warnings.append("单图无法观测围度，尺寸结果只能用于尺码分档，不可用于量体交付")

    return QualityReport(
        passed=not reasons,
        scale_cm_per_px=round(scale, 5),
        person_height_px=round(person_px, 1),
        reasons=reasons,
        warnings=warnings,
    )
