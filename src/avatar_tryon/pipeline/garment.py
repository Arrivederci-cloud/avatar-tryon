"""服装资产与形变目标插值。

设计取舍：运行时不做布料物理仿真。版型仿真在离线的 CLO/Style3D 里跑，
产出若干尺码档位的形变目标（morph target），运行时按体尺线性插值。

这样做的原因有两个：
1. 实时仿真在浏览器里不可能，成本也不可接受；
2. CLO 是商业软件，别人 clone 仓库跑不起来，离线预处理可以把依赖挡在仓库之外。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Garment:
    id: str
    name: str
    category: str
    anchors: list[dict]  # [{"chest": 88, "morph": {...}}, ...]


@dataclass(slots=True)
class MorphPlan:
    garment_id: str
    weights: dict[str, float]
    lower: float
    upper: float
    notes: list[str]


def load_catalog(path: Path) -> list[Garment]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Garment(
            id=g["id"],
            name=g["name"],
            category=g["category"],
            anchors=g["anchors"],
        )
        for g in raw["garments"]
    ]


def fit(garment: Garment, chest_cm: float) -> MorphPlan:
    """按胸围在相邻两个尺码档位之间线性插值。"""
    anchors = sorted(garment.anchors, key=lambda a: a["chest"])
    if not anchors:
        raise ValueError(f"{garment.id} 没有形变档位")

    if chest_cm <= anchors[0]["chest"]:
        a = anchors[0]
        return MorphPlan(garment.id, dict(a["morph"]), a["chest"], a["chest"],
                         ["低于最小档位，直接取最小档，可能出现松量偏大"])
    if chest_cm >= anchors[-1]["chest"]:
        a = anchors[-1]
        return MorphPlan(garment.id, dict(a["morph"]), a["chest"], a["chest"],
                         ["高于最大档位，直接取最大档，可能出现明显紧绷"])

    for lo, hi in zip(anchors, anchors[1:], strict=False):
        if lo["chest"] <= chest_cm <= hi["chest"]:
            t = (chest_cm - lo["chest"]) / (hi["chest"] - lo["chest"])
            keys = set(lo["morph"]) | set(hi["morph"])
            weights = {
                k: round(lo["morph"].get(k, 0.0) * (1 - t) + hi["morph"].get(k, 0.0) * t, 4)
                for k in keys
            }
            return MorphPlan(garment.id, weights, lo["chest"], hi["chest"],
                             [f"在 {lo['chest']} 与 {hi['chest']} 之间插值，t={t:.2f}"])

    raise RuntimeError("unreachable")
