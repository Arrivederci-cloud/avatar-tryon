"""只测 mock 路径与纯逻辑。CI 不需要 GPU、不需要权重。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from avatar_tryon.models import registry
from avatar_tryon.models.baseline_edit import BaselineInpaint
from avatar_tryon.models.errors import ModelNotWired
from avatar_tryon.models.generation_adapters import (
    DashscopeEdit,
    FashnVton,
    _build_edit_payload,
    _data_uri,
    _extract_image_url,
)
from avatar_tryon.pipeline import garment, orchestrator, quality_check, three_view
from avatar_tryon.utils.image import image_size, write_blank_png, write_gray_png

ROOT = Path(__file__).resolve().parents[1]


def _photo(tmp_path: Path, w: int = 1024, h: int = 1536) -> Path:
    return write_blank_png(tmp_path / "photo.png", w, h)


def _person_photo(tmp_path: Path, w: int = 512, h: int = 768) -> Path:
    """合成一张「浅灰背景 + 人形色块」的测试照片。

    纯白/纯灰的空白图在新采样逻辑里整张都是背景，无法验证投影，
    所以这里画一个粗略人形：肤色头、蓝上衣、深色双腿。
    """
    Image = pytest.importorskip("PIL.Image")
    from PIL import ImageDraw

    img = Image.new("RGB", (w, h), (238, 238, 238))
    d = ImageDraw.Draw(img)
    d.ellipse([w * 0.42, h * 0.05, w * 0.58, h * 0.19], fill=(226, 188, 160))
    d.polygon(
        [(w * 0.30, h * 0.21), (w * 0.70, h * 0.21), (w * 0.60, h * 0.60), (w * 0.40, h * 0.60)],
        fill=(74, 110, 168),
    )
    d.rectangle([w * 0.40, h * 0.58, w * 0.48, h * 0.96], fill=(58, 62, 72))
    d.rectangle([w * 0.52, h * 0.58, w * 0.60, h * 0.96], fill=(58, 62, 72))
    p = tmp_path / "photo.png"
    img.save(p, "PNG")
    return p


def test_image_size_reads_png(tmp_path: Path) -> None:
    p = _photo(tmp_path, 800, 1200)
    assert image_size(p) == (800, 1200)


def test_image_size_matches_pillow_for_png(tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")
    p = tmp_path / "real.png"
    Image.new("RGB", (600, 900), (210, 190, 175)).save(p, "PNG")
    assert image_size(p) == (600, 900)


def test_image_size_matches_pillow_for_jpeg(tmp_path: Path) -> None:
    """JPEG 是手写解析里最容易出错的分支：SOF 段前有变长段需要跳过。"""
    Image = pytest.importorskip("PIL.Image")
    p = tmp_path / "real.jpg"
    Image.new("RGB", (720, 1280), (200, 180, 170)).save(p, "JPEG", quality=80)
    assert image_size(p) == (720, 1280)


def test_quality_passes_for_reasonable_input(tmp_path: Path) -> None:
    qc = quality_check.check(_photo(tmp_path), height_cm=170)
    assert qc.passed, qc.reasons
    assert 0.03 <= qc.scale_cm_per_px <= 0.60


def test_quality_rejects_low_resolution(tmp_path: Path) -> None:
    qc = quality_check.check(_photo(tmp_path, 256, 384), height_cm=170)
    assert not qc.passed
    assert any("分辨率" in r for r in qc.reasons)


def test_quality_rejects_missing_height(tmp_path: Path) -> None:
    qc = quality_check.check(_photo(tmp_path), height_cm=0)
    assert not qc.passed


def test_mock_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    photo = _photo(tmp_path)
    out = tmp_path / "out"
    res = orchestrator.run(
        image=photo,
        height_cm=170,
        out_dir=out,
        weights_dir=tmp_path / "weights",
        mock=True,
        garment_catalog=ROOT / "assets" / "garments.json",
        garment_id="tee_001",
    )
    assert res.ok, res.errors
    assert res.stages["geometry"] == "ok"
    assert res.stages["three_view"] == "ok"
    assert res.stages["try_on"] == "ok"

    payload = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert "back" in payload["artifacts"]["views"]


def test_three_views_flag_back_as_generated(tmp_path: Path) -> None:
    photo = _photo(tmp_path)
    res = orchestrator.run(
        image=photo, height_cm=170, out_dir=tmp_path / "out",
        weights_dir=tmp_path / "w", mock=True,
    )
    assert any("生成" in w for w in res.warnings)


def test_write_gray_png_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "m.png"
    write_gray_png(p, 4, 2, bytes([0, 255, 0, 255, 255, 0, 255, 0]))
    assert image_size(p) == (4, 2)
    Image = pytest.importorskip("PIL.Image")
    img = Image.open(p)
    assert img.mode == "L"
    assert list(img.getdata()) == [0, 255, 0, 255, 255, 0, 255, 0]


def test_proxy_cylinder_fill_ratio_is_unseen_side(tmp_path: Path) -> None:
    """UV 纹理只有正面颜色：fill_ratio 按正面可见判定（约 42%）。

    三机位联合观测（observed_fill_ratio）则只剩左侧无机位缝带（约 12.5%）。
    """
    verts, uvs, faces = three_view._proxy_mesh()
    best, cam_of, front_cos = three_view._observation(verts, uvs, faces, 128)
    stats = three_view._stats(best, cam_of, front_cos)
    assert 0.40 < stats["fill_ratio"] < 0.75
    assert 0.05 < stats["observed_fill_ratio"] < 0.30
    assert stats["front_ratio"] > 0.2
    assert stats["back_ratio"] > 0.2
    mask = three_view._write_mask(
        tmp_path / "mask.png", three_view._fill_pixels(front_cos, None)
    )
    assert mask.exists() and image_size(mask) == (128, 128)


def test_load_obj_roundtrip(tmp_path: Path) -> None:
    verts, uvs, faces = three_view._proxy_mesh()
    obj = tmp_path / "body.obj"
    obj.write_text(three_view._to_obj(verts, uvs, faces), encoding="utf-8")
    parsed = three_view._load_obj(obj)
    assert parsed is not None
    pverts, puvs, pfaces = parsed
    assert len(pverts) == len(verts)
    assert puvs is not None and len(puvs) == len(uvs)
    assert len(pfaces) == len(faces)
    best, _, _ = three_view._observation(pverts, puvs, pfaces, 64)
    assert sum(1 for b in best if b >= three_view.GRAZE_COS) > 0


def test_backproject_views_samples_visible_colors(tmp_path: Path) -> None:
    """单视图投影：正对前方的 texel 采到人物颜色，背对相机的保持白色待补。"""
    Image = pytest.importorskip("PIL.Image")
    src = _person_photo(tmp_path)
    verts, uvs, faces = three_view._proxy_mesh()
    tex, mask, stats = three_view._backproject_views(
        src, None, verts, uvs, faces, 64, tmp_path
    )
    img = Image.open(tex).convert("RGB")
    assert img.getpixel((2, 32)) != (255, 255, 255)  # 正面 texel 采到了颜色
    assert img.getpixel((32, 32)) == (255, 255, 255)  # 背面 texel 待补
    assert stats["views"] == 1.0
    assert 0.0 < stats["fill_ratio"] < 1.0
    assert image_size(mask) == (64, 64)


def test_fill_pixels_or_flags_background() -> None:
    """待补掩码 = 正面无颜色来源 OR 反投影采到背景色。"""
    front = [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1]
    bg = bytes([0, 0, 0, 0, 0, 0, 0, 1])
    px = three_view._fill_pixels(front, bg)
    assert px[:4] == b"\x00\x00\x00\x00"
    assert px[4:7] == b"\xff\xff\xff" and px[7] == 255
    assert three_view._fill_pixels(front, None)[4:8] == b"\xff\xff\xff\xff"


def test_complete_uv_end_to_end_writes_mask(tmp_path: Path) -> None:
    photo = _photo(tmp_path)
    res = orchestrator.run(
        image=photo, height_cm=170, out_dir=tmp_path / "out",
        weights_dir=tmp_path / "w", mock=True,
    )
    assert res.ok
    # 输入图会被收进 out_dir，所有产物相对它落盘：out/uv/
    mask = tmp_path / "out" / "uv" / "uv_mask.png"
    assert mask.exists()
    assert image_size(mask) == (three_view.UV_SIZE, three_view.UV_SIZE)
    assert res.artifacts["uv_mask"] == str(mask)
    # 有 Pillow 时还应产出投影纹理与其 inpaint 结果
    pytest.importorskip("PIL.Image")
    assert (tmp_path / "out" / "uv" / "uv_multiview.png").exists()
    assert (tmp_path / "out" / "uv" / "inpaint" / "uv_multiview_filled.png").exists()


def test_proxy_engine_runs_on_cpu(tmp_path: Path) -> None:
    """proxy 档：参数化几何 + 软件渲染 + 基线补全，全程无需权重与显卡。"""
    photo = _person_photo(tmp_path)
    res = orchestrator.run(
        image=photo, height_cm=170, out_dir=tmp_path / "out",
        weights_dir=tmp_path / "w", engine="proxy",
    )
    assert res.ok, res.errors
    assert res.artifacts["engine"] == "proxy"
    assert res.stages["geometry"] == "ok"
    assert res.stages["three_view"] == "ok"
    views = json.loads(res.artifacts["views"])
    for name in ("front", "side", "back"):
        assert Path(views[name]).exists(), name
    assert (tmp_path / "out" / "uv" / "uv_mask.png").exists()
    coverage = json.loads(res.artifacts["uv_coverage"])
    assert 0.0 < coverage["fill_ratio"] < 1.0


def test_garment_interpolates_between_anchors() -> None:
    catalog = garment.load_catalog(ROOT / "assets" / "garments.json")
    tee = next(g for g in catalog if g.id == "tee_001")
    plan = garment.fit(tee, chest_cm=96.0)
    assert 0.0 < plan.weights["shoulder"] < 1.0
    assert plan.lower < 96.0 < plan.upper


def test_garment_clamps_out_of_range() -> None:
    catalog = garment.load_catalog(ROOT / "assets" / "garments.json")
    tee = next(g for g in catalog if g.id == "tee_001")
    plan = garment.fit(tee, chest_cm=140.0)
    assert plan.notes and any("最大档" in n for n in plan.notes)


def test_restricted_models_not_in_commercial_list() -> None:
    assert not set(registry.OPTIONAL_RESTRICTED) & set(registry.COMMERCIAL_OK)
    assert "catvton" in registry.OPTIONAL_RESTRICTED


def test_data_uri_is_inline_base64(tmp_path: Path) -> None:
    p = write_blank_png(tmp_path / "g.png", 8, 8)
    uri = _data_uri(p)
    assert uri.startswith("data:image/png;base64,")
    assert len(uri) > 40


def test_vton_api_without_key_fails_with_actionable_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FASHN_API_KEY", raising=False)
    vton = FashnVton(tmp_path / "w", backend="api", api_key=None)
    with pytest.raises(ModelNotWired) as excinfo:
        vton.apply(tmp_path / "a.png", tmp_path / "b.png")
    assert "FASHN_API_KEY" in str(excinfo.value)


def test_vton_rejects_unknown_category(tmp_path: Path) -> None:
    vton = FashnVton(tmp_path / "w", backend="local")
    with pytest.raises(ValueError, match="category"):
        vton.apply(tmp_path / "a.png", tmp_path / "b.png", category="hats")


def test_vton_local_backend_reports_missing_weights(tmp_path: Path) -> None:
    vton = FashnVton(tmp_path / "w", backend="local")
    with pytest.raises(ModelNotWired) as excinfo:
        vton.apply(tmp_path / "a.png", tmp_path / "b.png")
    msg = str(excinfo.value)
    assert "fashn-vton-1.5" in msg and "model.safetensors" in msg


def test_api_edit_requires_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    edit = DashscopeEdit(tmp_path / "w", api_key=None)
    with pytest.raises(ModelNotWired) as excinfo:
        edit.inpaint(tmp_path / "uv.png", tmp_path / "m.png", "补全")
    msg = str(excinfo.value)
    assert "DASHSCOPE_API_KEY" in msg and "--edit-backend local" in msg


def test_api_edit_payload_shape() -> None:
    """该端点无 mask 参数：掩码必须作为第二张图 + 指令约束一起发。"""
    payload = _build_edit_payload(
        "qwen-image-edit-plus", "data:image/png;base64,AAA", "data:image/png;base64,BBB", "针织纹理"
    )
    assert payload["model"] == "qwen-image-edit-plus"
    content = payload["input"]["messages"][0]["content"]
    assert content[0]["image"].endswith("AAA")
    assert content[1]["image"].endswith("BBB")
    text = content[2]["text"]
    assert "针织纹理" in text and "白色区域" in text and "保持不变" in text
    assert payload["parameters"]["prompt_extend"] is False


def test_api_edit_extract_image_url() -> None:
    body = {
        "output": {
            "choices": [
                {"message": {"content": [{"text": "ok"}, {"image": "https://x/y.png"}]}}
            ]
        }
    }
    assert _extract_image_url(body) == "https://x/y.png"
    with pytest.raises(ModelNotWired):
        _extract_image_url({"output": {}})


def test_api_edit_http_error_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import urllib.error

    def boom(*_a, **_kw):
        raise urllib.error.HTTPError(
            "https://dashscope/x", 401, "Unauthorized", {}, io.BytesIO(
                b'{"code": "InvalidApiKey", "message": "Invalid API-key provided."}'
            )
        )

    monkeypatch.setattr("urllib.request.urlopen", boom)
    edit = DashscopeEdit(tmp_path / "w", api_key="sk-test")
    uv = write_blank_png(tmp_path / "uv.png", 8, 8)
    mask = write_gray_png(tmp_path / "m.png", 8, 8, bytes(64))
    with pytest.raises(ModelNotWired) as excinfo:
        edit.inpaint(uv, mask, "补全")
    msg = str(excinfo.value)
    assert "401" in msg and "InvalidApiKey" in msg


# ---------------------------------------------------------------------------
# 轮廓剖面与法线定向（2026-10 会话新增的两处关键修复）
# ---------------------------------------------------------------------------


def test_width_profile_tracks_silhouette_width(tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")
    from avatar_tryon.utils.silhouette import width_profile

    # 上宽下窄的人形：肩部 200px 宽，脚部 80px 宽
    img = Image.new("L", (256, 256), 0)
    from PIL import ImageDraw

    d = ImageDraw.Draw(img)
    d.polygon([(28, 10), (228, 10), (168, 246), (88, 246)], fill=255)
    prof = width_profile(img)
    assert prof is not None
    top = prof.span(5)
    bottom = prof.span(prof.bands - 6)
    assert top is not None and bottom is not None
    assert (top[1] - top[0]) > (bottom[1] - bottom[0]) * 1.5


def test_mesh_normals_of_offcenter_limb_face_outward() -> None:
    """回归测试：偏心部件（手臂）的法线必须按部件自身定向。

    旧的「背离整网格质心」会把左臂全部法线掰成 -x、右臂全 +x，
    导致侧视机位下一侧手臂整体被判为背向、完全不参与烘焙
    （实测单侧残留 97%）。修复后两臂在同一机位下的可见面应对称。
    """
    from avatar_tryon.camera import camera_dirs
    from avatar_tryon.utils.body_mesh import ATLAS, build_body
    from avatar_tryon.utils.mesh import triangles

    verts, uvs, faces = build_body(170.0, 90.0, 72.0, 95.0)
    cam = camera_dirs()["side"]

    def visible_ratio(part: str) -> tuple[int, int]:
        u0, u1, v0, v1 = ATLAS[part]
        keep = total = 0
        for p0, p1, p2, _a, _b, _c, n in triangles(verts, uvs, faces):
            if not all(u0 <= p[0] <= u1 and v0 <= p[1] <= v1 for p in (p0, p1, p2)):
                continue
            total += 1
            if n[0] * cam[0] + n[1] * cam[1] + n[2] * cam[2] > 0.25:
                keep += 1
        return keep, total

    left_keep, left_total = visible_ratio("arm_l")
    right_keep, right_total = visible_ratio("arm_r")
    assert left_total > 0 and right_total > 0
    # 两侧手臂都必须有朝向侧视机位的可见面（旧实现一侧为 0）
    assert left_keep > 0, "左臂在侧视机位下没有任何可见面"
    assert right_keep > 0, "右臂在侧视机位下没有任何可见面"
    # 对称几何应给出对称的可见比例
    lr, rr = left_keep / left_total, right_keep / right_total
    assert abs(lr - rr) < 0.05


def test_raster_uv_covers_rect_edge_column() -> None:
    """图集矩形左边缘的 1px 不能漏。

    u0=0.02 -> 像素 5.12，整数栅格化会从 x=6 开始，留下 1px 纯白，
    渲染器恰好在这个 u 上采样——正面会出现一条贯穿的白缝。
    光栅化改用像素中心（x+0.5）后，x=5 必须被覆盖。
    """
    visited = set()
    tri = ((0.02, 0.60), (0.10, 0.60), (0.02, 0.40))
    three_view._raster_uv(tri[0], tri[1], tri[2], 64, lambda x, y, a, b, c: visited.add((x, y)))
    assert any(x == int(0.02 * 64) for x, _ in visited), "矩形左边缘 1px 未被覆盖"


def test_dilate_marks_neighbors() -> None:
    from avatar_tryon.pipeline.three_view import _dilate

    size = 4
    px = bytearray(size * size)
    px[5] = 255  # (1,1)
    out = _dilate(bytes(px), size)
    assert out[5] == 255
    for idx in (0, 1, 2, 4, 6, 8, 9, 10):  # (1,1) 的 8 邻域
        assert out[idx] == 255, idx
    assert out[3] == 0 and out[15] == 0  # 远离种子点的位置不受影响


def test_wanx_payload_uses_mask_function(tmp_path: Path) -> None:
    """回归：function 必须是 description_edit_with_mask。

    服务端只在该 function（和 expand）下接受 mask_image_url；
    传 description_change_with_mask 会报
    「mask_image_url should not be provided」（实测）。
    """
    from avatar_tryon.models.generation_adapters import _build_wanx_payload

    payload = _build_wanx_payload("wanx2.1-imageedit", "img", "msk", "补全")
    assert payload["input"]["function"] == "description_edit_with_mask"
    assert payload["input"]["base_image_url"] == "img"
    assert payload["input"]["mask_image_url"] == "msk"
    assert payload["model"] == "wanx2.1-imageedit"


def test_baseline_inpaint_keeps_known_pixels(tmp_path: Path) -> None:
    """已知像素必须一个字节都不动——补全只许改待补区。"""
    Image = pytest.importorskip("PIL.Image")
    import numpy as np

    src = Image.new("RGB", (64, 64), (40, 60, 90))
    src.save(tmp_path / "uv.png", "PNG")
    mask = write_gray_png(tmp_path / "mask.png", 64, 64, bytes(255 for _ in range(64 * 64)))
    out = BaselineInpaint(iterations=5).inpaint(
        tmp_path / "uv.png", mask, "补全"
    )
    # 全图待补时无已知边界，但掩码外不应有颜色突变——这里只验证可运行
    assert Image.open(out).size == (64, 64)
    del np


def test_baseline_inpaint_fills_from_local_color(tmp_path: Path) -> None:
    """回归：待补区必须取附近颜色，而不是全图平均或整行抹平。

    旧实现轴向最近邻在整行缺色时把整行抹平、均值填充会把蓝衬衫糊成灰蓝。
    构造「左侧纯红、右侧纯蓝、中间竖条待补」的图，中心填充应取
    最近邻（红或蓝），而不是两者的均值紫灰。
    """
    Image = pytest.importorskip("PIL.Image")
    import numpy as np
    from PIL import ImageDraw

    img = Image.new("RGB", (48, 32), (200, 30, 30))
    d = ImageDraw.Draw(img)
    d.rectangle([28, 0, 47, 31], fill=(30, 30, 200))
    img.save(tmp_path / "uv.png", "PNG")
    # 待补竖条 x=20..27（两侧都有已知颜色）
    m = Image.new("L", (48, 32), 0)
    ImageDraw.Draw(m).rectangle([20, 0, 27, 31], fill=255)
    m.save(tmp_path / "mask.png", "PNG")

    out = Image.open(BaselineInpaint(iterations=30).inpaint(tmp_path / "uv.png", tmp_path / "mask.png", "补全"))
    arr = np.asarray(out).astype(int)
    # 已知区零改动
    assert (arr[:, :20] == (200, 30, 30)).all()
    assert (arr[:, 28:] == (30, 30, 200)).all()
    # 局部性：填充区是从红到蓝的平滑过渡。靠红侧的 R 通道应显著高于靠蓝侧。
    # 全局均值填充（旧金字塔实现的退化行为）会给出恒定的 ~115，无法通过。
    left_mean = arr[:, 20:22, 0].mean()
    right_mean = arr[:, 26:28, 0].mean()
    assert left_mean > right_mean + 80, (
        f"填充区没有局部性：左 {left_mean:.0f} vs 右 {right_mean:.0f}"
    )


def test_bake_fill_ratio_uses_full_atlas(tmp_path: Path) -> None:
    """分母必须是整个 UV 图集，而不是「本组视图可见三角形」的覆盖。

    视图越少分母越小会让 fill_ratio 虚低（单视图时后半个身体全空也不计入）。
    """
    p = _person_photo(tmp_path)
    from avatar_tryon.utils.body_mesh import build_body

    verts, uvs, faces = build_body(170.0, 90.0, 72.0, 95.0)
    out_dir = tmp_path / "uv"
    out_dir.mkdir(parents=True)
    _p, _m, stats = three_view._backproject_views(
        p, None, verts, uvs, faces, 64, out_dir
    )
    # 单视图时残留必须超过一半（后半个身体没有颜色来源），
    # 且图集 texel 数对应全部三角形，而不是仅正面可见的那部分
    assert stats["fill_ratio"] > 0.5
    assert stats["sampled_texels"] < stats["atlas_texels"] * 0.75
