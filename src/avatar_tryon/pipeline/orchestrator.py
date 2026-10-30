"""流水线编排。

顺序：质检与标定 -> 几何 -> 三视图 -> 绑定 -> （可选）试衣/试发。

两个工程约束写在这里，而不是散落在各模块：
1. 任何一步失败都要产出可读的中间结果，不能静默吞掉；
2. LLM 不在这条链路上。搭配建议是独立的后置步骤，LLM 挂了主流程照常完成。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..models import registry
from ..models.errors import ModelNotWired
from . import garment as garment_mod
from . import quality_check, three_view


@dataclass(slots=True)
class PipelineResult:
    ok: bool
    out_dir: Path
    stages: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def run(
    image: Path,
    height_cm: float,
    out_dir: Path,
    weights_dir: Path,
    mock: bool = False,
    garment_catalog: Path | None = None,
    garment_id: str | None = None,
    garment_image: Path | None = None,
    vton_backend: str = "api",
    vton_category: str = "tops",
    hairstyle: Path | None = None,
    engine: str | None = None,
    edit_backend: str = "api",
    views_backend: str = "local",
) -> PipelineResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    res = PipelineResult(ok=True, out_dir=out_dir)
    engine = engine or ("mock" if mock else "real")
    res.artifacts["engine"] = engine
    if engine == "proxy":
        res.warnings.append(
            "proxy 档输出为链路验证预览（代理几何 + 基线补全），不是产品效果，请勿用于展示或交付"
        )

    # 各阶段产物按「图片所在目录」组织，所以先把输入图收进 out_dir，
    # 保证 --out 指定的目录拥有全部产物，而不是散落在原图旁。
    if image.resolve().parent != out_dir.resolve():
        local = out_dir / image.name
        shutil.copy2(image, local)
        image = local

    # 1. 质检与尺度
    qc = quality_check.check(image, height_cm)
    res.stages["quality"] = "passed" if qc.passed else "failed"
    res.warnings.extend(qc.warnings)
    if not qc.passed:
        res.ok = False
        res.errors.extend(qc.reasons)
        _dump(out_dir, res, qc)
        return res
    res.artifacts["scale_cm_per_px"] = str(qc.scale_cm_per_px)

    # 2. 几何
    try:
        recon = registry.build_recon("lhm", weights_dir, engine)
        geom = recon.reconstruct(image, height_cm)
        res.stages["geometry"] = "ok"
        res.artifacts["mesh"] = str(geom.mesh_path)
        res.artifacts["measurements"] = json.dumps(geom.measurements, ensure_ascii=False)
    except ModelNotWired as exc:
        res.ok = False
        res.stages["geometry"] = "not_wired"
        res.errors.append(str(exc))
        _dump(out_dir, res, qc)
        return res

    # 3. 三视图
    try:
        views_model = registry.build_views("lhm", weights_dir, engine, backend=views_backend)
        edit_model = registry.build_edit("qwen_image_edit", weights_dir, engine, backend=edit_backend)
        bundle = three_view.build(image, geom, views_model, edit_model, fill_uv=True)
        res.stages["three_view"] = "ok"
        res.artifacts["views"] = json.dumps(
            {k: str(v) for k, v in asdict(bundle.views).items() if isinstance(v, Path)},
            ensure_ascii=False,
        )
        if bundle.uv_mask is not None:
            res.artifacts["uv_mask"] = str(bundle.uv_mask)
        if bundle.uv_texture is not None:
            res.artifacts["uv_texture"] = str(bundle.uv_texture)
        if bundle.coverage:
            res.artifacts["uv_coverage"] = json.dumps(bundle.coverage, ensure_ascii=False)
        res.warnings.extend(bundle.notes)
    except ModelNotWired as exc:
        res.ok = False
        res.stages["three_view"] = "not_wired"
        res.errors.append(str(exc))
        _dump(out_dir, res, qc)
        return res

    # 4. 绑定
    try:
        rig_model = registry.build_rig("unirig", weights_dir, engine)
        avatar = rig_model.rig(geom.mesh_path)
        res.stages["rig"] = "ok"
        res.artifacts["avatar"] = str(avatar.mesh_path)
    except ModelNotWired as exc:
        res.stages["rig"] = "not_wired"
        res.warnings.append(str(exc))

    # 5. 可选：试衣
    if garment_catalog and garment_id:
        catalog = garment_mod.load_catalog(garment_catalog)
        target = next((g for g in catalog if g.id == garment_id), None)
        if target is None:
            res.warnings.append(f"目录中找不到款式 {garment_id}")
        else:
            plan = garment_mod.fit(target, geom.measurements.get("chest", 88.0))
            res.artifacts["morph"] = json.dumps(asdict(plan), ensure_ascii=False)
            res.warnings.extend(plan.notes)

            if garment_image is None and engine != "mock":
                res.stages["try_on"] = "skipped"
                res.warnings.append("未提供 --garment-image，跳过试穿；形变目标已计算")
            else:
                try:
                    vton = registry.build_vton(
                        "fashn_vton", weights_dir, engine, backend=vton_backend
                    )
                    out = vton.apply(
                        bundle.views.front,
                        garment_image or image,
                        garment_id,
                        category=vton_category,
                    )
                    res.stages["try_on"] = "ok"
                    res.artifacts["try_on"] = str(out.image_path)
                except ModelNotWired as exc:
                    res.stages["try_on"] = "not_wired"
                    res.warnings.append(str(exc))

    # 6. 可选：试发
    if hairstyle:
        try:
            hair = registry.build_hair("hairfastgan", weights_dir, engine)
            p = hair.transfer(bundle.views.front, hairstyle)
            res.stages["hairstyle"] = "ok"
            res.artifacts["hairstyle"] = str(p)
        except ModelNotWired as exc:
            res.stages["hairstyle"] = "not_wired"
            res.warnings.append(str(exc))

    _dump(out_dir, res, qc)
    return res


def _dump(out_dir: Path, res: PipelineResult, qc) -> Path:
    payload = {
        "ok": res.ok,
        "stages": res.stages,
        "artifacts": res.artifacts,
        "warnings": res.warnings,
        "errors": res.errors,
        "quality": asdict(qc),
    }
    path = out_dir / "result.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
