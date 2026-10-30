"""Mock 实现：不加载任何模型，产出占位文件。

存在的意义是让任何人 clone 下来就能跑通整条流水线，
验证接口与数据流是否正确，而不必先准备好显卡和几十 GB 权重。
CI 里也只跑 mock。
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from .base import (
    Avatar,
    EditModel,
    HairModel,
    HumanGeometry,
    ReconModel,
    RigModel,
    ThreeViews,
    TryOnModel,
    TryOnResult,
    ViewModel,
)

# 1x1 透明 PNG
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _write_png(path: Path, tag: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_PNG)
    path.with_suffix(".txt").write_text(f"mock placeholder: {tag}\n", encoding="utf-8")
    return path


class MockRecon(ReconModel):
    name = "mock-recon"
    license = "n/a"

    def reconstruct(self, image: Path, height_cm: float) -> HumanGeometry:
        out = image.parent / "geometry"
        mesh = out / "body.obj"
        mesh.parent.mkdir(parents=True, exist_ok=True)
        mesh.write_text("# mock mesh\n", encoding="utf-8")
        h = float(height_cm)
        measurements = {
            "height": h,
            "chest": round(h * 0.53, 1),
            "waist": round(h * 0.42, 1),
            "hip": round(h * 0.56, 1),
        }
        return HumanGeometry(
            mesh_path=mesh,
            measurements=measurements,
            joints_cm={"head": (0.0, 0.0, h), "hip": (0.0, 0.0, h * 0.53)},
            scale_cm_per_unit=1.0,
            notes=["mock geometry, 数值由身高按经验比例推算，仅供流水线联调"],
        )


class MockViews(ViewModel):
    name = "mock-views"
    license = "n/a"

    def render_views(self, image: Path, geometry: HumanGeometry) -> ThreeViews:
        out = image.parent / "views"
        return ThreeViews(
            front=_write_png(out / "front.png", "front"),
            side=_write_png(out / "side.png", "side"),
            back=_write_png(out / "back.png", "back"),
            confidence={"front": 1.0, "side": 0.5, "back": 0.2},
            notes=["back view 完全由先验生成，不是观测结果"],
        )


class MockEdit(EditModel):
    name = "mock-edit"
    license = "n/a"

    def inpaint(self, image: Path, mask: Path, prompt: str) -> Path:
        out = image.parent / "inpaint" / f"{image.stem}_filled.png"
        _write_png(out, f"inpaint: {prompt}")
        return out


class MockTryOn(TryOnModel):
    name = "mock-vton"
    license = "n/a"

    def apply(
        self, person: Path, garment: Path, garment_id: str = "demo", **kwargs: object
    ) -> TryOnResult:
        out = _write_png(person.parent / "tryon" / f"{garment_id}.png", f"tryon {garment_id}")
        return TryOnResult(image_path=out, garment_id=garment_id, notes=["mock try-on"])


class MockHair(HairModel):
    name = "mock-hair"
    license = "n/a"

    def transfer(self, person: Path, hairstyle: Path, color: str | None = None) -> Path:
        return _write_png(person.parent / "hair" / f"hair_{color or 'default'}.png", "hair")


class MockRig(RigModel):
    name = "mock-rig"
    license = "n/a"

    def rig(self, mesh_path: Path) -> Avatar:
        rigged = mesh_path.with_name(mesh_path.stem + "_rigged.glb")
        rigged.write_text("glTF mock rig\n", encoding="utf-8")
        return Avatar(mesh_path=rigged, texture_path=None, rig_path=rigged, notes=["mock rig"])


class MockManifest:
    """把 mock 结果落成 json，方便人工核对数据流。"""

    @staticmethod
    def dump(path: Path, payload: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
