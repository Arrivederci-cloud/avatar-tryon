"""骨骼绑定适配器：UniRig（MIT）。"""

from __future__ import annotations

from pathlib import Path

from .base import Avatar, RigModel
from .errors import ModelNotWired


class UniRig(RigModel):
    name = "unirig"
    license = "MIT"
    upstream = "https://github.com/VAST-AI-Research/UniRig"

    def __init__(self, weights_dir: Path) -> None:
        self.weights_dir = weights_dir

    def rig(self, mesh_path: Path) -> Avatar:
        if not (self.weights_dir / "unirig").exists():
            raise ModelNotWired(
                f"未找到 UniRig 权重：{self.weights_dir / 'unirig'}\n"
                f"先运行 python scripts/fetch_weights.py --model unirig"
            )
        raise ModelNotWired(
            "接入要点：输入需为单一水密网格、T-pose 或 A-pose、脚底为原点。"
            "UniRig 输出的是重心原点且无骨骼语义命名，接引擎前需要一次规范化："
            "把 hips 移到脚底、按 Mixamo 命名重写骨骼，否则动画重定向会错位。"
        )
