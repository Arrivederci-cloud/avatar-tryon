"""候选对比的测试。

重点不是排版像素级正确，而是三条行为约定：
1. 类型只按文件名前缀判断，不做内容猜测；
2. 单个候选失败（缺权重/缺 key/坏图）不许拖垮整轮比较；
3. 报告必须如实记录每个候选的状态，不能只报成功的。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from avatar_tryon.models.errors import ModelNotWired
from avatar_tryon.models.mock import MockHair, MockTryOn
from avatar_tryon.pipeline import compare


def _png(path: Path, color=(120, 140, 160), size=(96, 128)) -> Path:
    Image = pytest.importorskip("PIL.Image")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


# ---------- 类型判定 ----------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("hair_bob.png", compare.KIND_HAIR),
        ("hairstyle-01.jpg", compare.KIND_HAIR),
        ("WORN_look.png", compare.KIND_WORN),
        ("ref_celebrity.jpg", compare.KIND_WORN),
        ("tee.jpg", compare.KIND_GARMENT),
        ("jacket_blue.png", compare.KIND_GARMENT),
        ("no_prefix_at_all.jpg", compare.KIND_GARMENT),
    ],
)
def test_infer_kind_by_filename_prefix(name: str, expected: str) -> None:
    assert compare.infer_kind(Path(name)) == expected


def test_discover_expands_directory_in_stable_order(tmp_path: Path) -> None:
    for name in ("b_tee.jpg", "a_coat.jpg", "c_dress.png"):
        _png(tmp_path / "cands" / name)
    (tmp_path / "cands" / "notes.txt").write_text("不是图片", encoding="utf-8")

    cands = compare.discover([tmp_path / "cands"])
    assert [c.id for c in cands] == ["a_coat", "b_tee", "c_dress"]
    assert all(c.kind == compare.KIND_GARMENT for c in cands)


def test_discover_deduplicates_ids(tmp_path: Path) -> None:
    """两张不同目录下的同名图不能互相覆盖产物。"""
    _png(tmp_path / "odd" / "tee.jpg")
    _png(tmp_path / "even" / "tee.jpg")
    cands = compare.discover([tmp_path / "odd" / "tee.jpg", tmp_path / "even" / "tee.jpg"])
    assert [c.id for c in cands] == ["tee", "tee_2"]
    assert len({c.id for c in cands}) == 2


def test_discover_rejects_non_image(tmp_path: Path) -> None:
    bad = tmp_path / "readme.md"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        compare.discover([bad])


# ---------- 失败隔离 ----------


def test_missing_vton_records_unwired_but_keeps_other_candidates(tmp_path: Path) -> None:
    person = _png(tmp_path / "person.png")
    cands = compare.discover([_png(tmp_path / "tee.png"), _png(tmp_path / "hair_bob.png")])

    report = compare.run(person, cands, tmp_path / "cmp", vton=None, hair=MockHair())

    by_id = {o.candidate.id: o for o in report.outcomes}
    assert by_id["tee"].status == compare.STATUS_UNWIRED
    assert "试穿模型" in by_id["tee"].note
    # 发型那件照常完成——一个后端缺失不该让整轮比较失效
    assert by_id["hair_bob"].status == compare.STATUS_OK
    assert report.ok_count == 1


def test_backend_exception_is_recorded_not_raised(tmp_path: Path) -> None:
    person = _png(tmp_path / "person.png")
    cands = compare.discover([_png(tmp_path / "tee.png")])

    class Boom(MockTryOn):
        def apply(self, *a, **kw):
            raise ModelNotWired("权重缺失：weights/fashn_vton/model.safetensors")

    report = compare.run(person, cands, tmp_path / "cmp", vton=Boom(), hair=MockHair())
    assert report.outcomes[0].status == compare.STATUS_UNWIRED
    assert "权重缺失" in report.outcomes[0].note


def test_worn_reference_is_registered_not_rendered(tmp_path: Path) -> None:
    person = _png(tmp_path / "person.png")
    cands = compare.discover([_png(tmp_path / "worn_look.png")])
    report = compare.run(person, cands, tmp_path / "cmp", vton=MockTryOn(), hair=MockHair())
    outcome = report.outcomes[0]
    assert outcome.status == compare.STATUS_REFERENCE
    assert outcome.output == tmp_path / "worn_look.png"  # 原图，不是新渲染的产物


# ---------- 产物 ----------


def test_run_writes_sheet_and_json(tmp_path: Path) -> None:
    pytest.importorskip("PIL.Image")
    person = _png(tmp_path / "person.png")
    cands = compare.discover(
        [_png(tmp_path / "tee.png"), _png(tmp_path / "hair_bob.png"), _png(tmp_path / "worn_x.png")]
    )
    report = compare.run(
        person, cands, tmp_path / "cmp", vton=MockTryOn(), hair=MockHair(), cols=2
    )

    assert report.sheet is not None and report.sheet.exists()
    assert (tmp_path / "cmp" / "compare.json").exists()
    # 每个成功候选都应有一份收进对比目录的副本
    assert (tmp_path / "cmp" / "tee.png").exists()
    assert (tmp_path / "cmp" / "hair_bob.png").exists()


def test_sheet_grid_geometry(tmp_path: Path) -> None:
    pytest.importorskip("PIL.Image")
    from PIL import Image

    person = _png(tmp_path / "person.png")
    cands = compare.discover([_png(tmp_path / "tee.png")])
    report = compare.run(person, cands, tmp_path / "cmp", vton=MockTryOn(), hair=MockHair())

    with Image.open(report.sheet) as sheet:
        # 2 格（原图 + 1 候选）单行
        assert sheet.size == (
            2 * compare.CELL_W + 3 * compare.GAP,
            compare.CELL_H + compare.LABEL_H + 2 * compare.GAP,
        )


def test_sheet_skipped_without_pillow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """无 Pillow 时不能崩：产物照旧，图层跳过并留下说明。"""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("PIL"):
            raise ImportError("simulated: no pillow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    person = tmp_path / "person.png"
    person.write_bytes(b"\x89PNG\r\n\x1a\n")  # 只要存在即可，mock 后端不读它
    cand = compare.Candidate(id="tee", kind=compare.KIND_GARMENT, path=person)
    report = compare.run(person, [cand], tmp_path / "cmp", vton=MockTryOn(), hair=MockHair())

    assert report.sheet is None
    assert any("Pillow" in n for n in report.notes)


# ---------- 试穿类别 ----------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("tee.jpg", "tops"),
        ("blue_jeans.jpg", "bottoms"),
        ("wide_trouser.png", "bottoms"),
        ("summer-dress.jpg", "one-pieces"),
        ("hoodie_02.png", "tops"),
        ("img_0001.jpg", None),
    ],
)
def test_infer_category_from_filename(name: str, expected: str | None) -> None:
    assert compare.infer_category(Path(name)) == expected


def test_resolve_category_auto_marks_the_fallback() -> None:
    """猜不出时要用默认值，但必须自报家门——不能让用户以为识别成功了。"""
    cand = compare.Candidate(
        id="img_0001", kind=compare.KIND_GARMENT, path=Path("img_0001.jpg")
    )
    value, note = compare.resolve_category(cand, compare.CATEGORY_AUTO)
    assert value == compare.CATEGORY_DEFAULT
    assert "未识别" in note


def test_resolve_category_explicit_wins_over_filename() -> None:
    cand = compare.Candidate(id="x", kind=compare.KIND_GARMENT, path=Path("jeans.jpg"))
    value, note = compare.resolve_category(cand, "one-pieces")
    assert value == "one-pieces"
    assert note == "category=one-pieces"


def test_dispatch_uses_each_candidates_own_category(tmp_path: Path) -> None:
    """一批混着上衣和裤子时，每件必须拿自己的类别去试穿，不能整批一刀切。"""
    person = _png(tmp_path / "person.png")
    seen: dict[str, object] = {}

    class Recorder(MockTryOn):
        def apply(self, person, garment, garment_id="demo", **kw):
            seen[garment_id] = kw.get("category")
            return super().apply(person, garment, garment_id, **kw)

    cands = compare.discover([_png(tmp_path / "tee.png"), _png(tmp_path / "jeans.png")])
    compare.run(person, cands, tmp_path / "cmp", vton=Recorder(), hair=MockHair())
    assert seen == {"tee": "tops", "jeans": "bottoms"}


def test_run_rejects_unknown_category(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="category"):
        compare.run(tmp_path / "p.png", [], tmp_path / "cmp", category="shoes")


# ---------- 报告 ----------


def test_table_lists_every_candidate(tmp_path: Path) -> None:
    person = _png(tmp_path / "person.png")
    cands = compare.discover([_png(tmp_path / "tee.png"), _png(tmp_path / "worn_x.png")])
    report = compare.run(person, cands, tmp_path / "cmp", vton=MockTryOn(), hair=MockHair())
    text = report.table()
    assert "tee" in text and "worn_x" in text
    assert "reference" in text


def test_report_dict_is_json_serialisable(tmp_path: Path) -> None:
    import json

    person = _png(tmp_path / "person.png")
    cands = compare.discover([_png(tmp_path / "tee.png")])
    report = compare.run(person, cands, tmp_path / "cmp", vton=MockTryOn(), hair=MockHair())
    data = json.loads(json.dumps(report.as_dict(), ensure_ascii=False))
    assert data["total"] == 1
    assert data["outcomes"][0]["kind"] == compare.KIND_GARMENT
