"""模型适配器接口。

设计原则：上层流水线只依赖这里的抽象，不依赖任何具体模型实现。
这样非自由许可的模型（如 SAM 3D Body）可以只作为可选实现存在，
仓库主体永远保持可分发。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class HumanGeometry:
    """参数化人体几何。单位统一为厘米。"""

    mesh_path: Path
    measurements: dict[str, float] = field(default_factory=dict)
    joints_cm: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    scale_cm_per_unit: float = 1.0
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ThreeViews:
    """正、侧、背三视图。中间表示，供人工确认。"""

    front: Path
    side: Path
    back: Path
    confidence: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TryOnResult:
    image_path: Path
    garment_id: str
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Avatar:
    """可驱动的三维数字人。"""

    mesh_path: Path
    texture_path: Path | None
    rig_path: Path | None
    notes: list[str] = field(default_factory=list)


class ReconModel(ABC):
    """单图或单图加尺度先验 -> 人体几何。"""

    name: str = "unknown"
    license: str = "unknown"

    @abstractmethod
    def reconstruct(self, image: Path, height_cm: float) -> HumanGeometry:
        raise NotImplementedError


class ViewModel(ABC):
    """几何 + 原图 -> 正侧背三视图。"""

    name: str = "unknown"
    license: str = "unknown"

    @abstractmethod
    def render_views(self, image: Path, geometry: HumanGeometry) -> ThreeViews:
        raise NotImplementedError


class EditModel(ABC):
    """图像局部重绘。用于 UV 空间的纹理补全。"""

    name: str = "unknown"
    license: str = "unknown"

    @abstractmethod
    def inpaint(self, image: Path, mask: Path, prompt: str) -> Path:
        raise NotImplementedError


class TryOnModel(ABC):
    """2D 服装试穿。"""

    name: str = "unknown"
    license: str = "unknown"

    @abstractmethod
    def apply(self, person: Path, garment: Path, garment_id: str = "demo") -> TryOnResult:
        raise NotImplementedError


class HairModel(ABC):
    """2D 发型迁移。"""

    name: str = "unknown"
    license: str = "unknown"

    @abstractmethod
    def transfer(self, person: Path, hairstyle: Path, color: str | None = None) -> Path:
        raise NotImplementedError


class RigModel(ABC):
    """静态网格 -> 带蒙皮权重的可驱动模型。"""

    name: str = "unknown"
    license: str = "unknown"

    @abstractmethod
    def rig(self, mesh_path: Path) -> Avatar:
        raise NotImplementedError


__all__ = [
    "Avatar",
    "EditModel",
    "HairModel",
    "HumanGeometry",
    "ReconModel",
    "RigModel",
    "ThreeViews",
    "TryOnModel",
    "ViewModel",
]
