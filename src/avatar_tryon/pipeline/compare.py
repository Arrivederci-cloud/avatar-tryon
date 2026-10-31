"""候选对比：把「衣服 / 发型 / 模特上身图」批量渲染出来并排摆放，供人挑选。

为什么要有这个模块：试穿一次只能回答「这件行不行」，而用户真实的问题是
「这几件里哪件更适合我」。后者需要并排看——分别打开十张图、来回切换，
人是比较不出来的，也会记住第一张而不是最好的一张。

三件约定：

1. **候选类型由文件名前缀决定，不猜内容**。
   `hair_*` / `hairstyle_*` → 发型；`worn_*` / `ref_*` / `look_*` → 模特上身图；
   其余一律当衣服。猜图片内容是模型的活，猜错了用户比的是错的东西，
   所以宁可要求改名，也不要静默误判。

2. **模特上身图只作参照，不做试穿**。它是「我想要的效果」，
   不是可试穿的资产——从模特身上抠出那件衣服需要服饰分割模型，
   本项目默认不带（aitryon-parsing 需账号开通）。
   直接把它当衣服喂给试穿，出来的是「把模特整张图当布料」，纯属浪费算力。

3. **单个候选失败不影响整张对比图**。缺权重、缺 key、图坏了，
   都记成一行状态而不是抛异常——否则用户会因为第三件衣服报错而看不到前两件。

4. **每件衣服的试穿类别随候选走，不整批一刀切**。一批里混着上衣和裤子，
   统一按 `tops` 喂会把裤子按上半身重绘，结果全错。默认 `auto` 按文件名
   关键词判断，判断结果写进表格；判断不出就落回默认值并注明。

诚实的边界：本模块不产出「推荐指数」。换装好不好看是审美判断，
拿像素差之类的代理指标排序会给出看似科学、实则误导的结论。
它只负责把候选摆齐、把客观信息（状态、耗时、与原图差异）标清楚，
**排序交给人**。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..models.base import HairModel, TryOnModel
from ..models.errors import ModelNotWired

KIND_GARMENT = "garment"
KIND_HAIR = "hairstyle"
KIND_WORN = "worn"
KINDS = (KIND_GARMENT, KIND_HAIR, KIND_WORN)

# 试穿接口的 category 单值参数 + auto（逐候选按文件名判断，见 CATEGORY_KEYWORDS）。
CATEGORIES = ("tops", "bottoms", "one-pieces")
CATEGORY_AUTO = "auto"
CATEGORY_DEFAULT = "tops"

# 文件名关键词 → 试穿类别。仅在 category=auto 时启用。
# 为什么需要：一批候选里同时有上衣和裤子是常见诉求（「这几件配起来怎么样」），
# 而 category 是单值参数，统一按 tops 喂裤子会让人像被按上半身重绘，结果全错。
# 猜不中时落回 CATEGORY_DEFAULT，并且表格里写明是默认值，不静默。
CATEGORY_KEYWORDS: dict[str, str] = {
    "top": "tops",
    "tops": "tops",
    "tee": "tops",
    "shirt": "tops",
    "blouse": "tops",
    "sweater": "tops",
    "hoodie": "tops",
    "jacket": "tops",
    "coat": "tops",
    "cardigan": "tops",
    "knit": "tops",
    "bottom": "bottoms",
    "bottoms": "bottoms",
    "pant": "bottoms",
    "pants": "bottoms",
    "jean": "bottoms",
    "jeans": "bottoms",
    "trouser": "bottoms",
    "trousers": "bottoms",
    "skirt": "bottoms",
    "short": "bottoms",
    "shorts": "bottoms",
    "dress": "one-pieces",
    "onepiece": "one-pieces",
    "jumpsuit": "one-pieces",
    "robe": "one-pieces",
    "gown": "one-pieces",
}

# 前缀 → 类型。只认文件名，不做内容识别。
PREFIX_TO_KIND: dict[str, str] = {
    "hair": KIND_HAIR,
    "hairstyle": KIND_HAIR,
    "haircut": KIND_HAIR,
    "worn": KIND_WORN,
    "ref": KIND_WORN,
    "look": KIND_WORN,
    "model": KIND_WORN,
    "garment": KIND_GARMENT,
    "top": KIND_GARMENT,
    "bottom": KIND_GARMENT,
    "dress": KIND_GARMENT,
    "coat": KIND_GARMENT,
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

STATUS_OK = "ok"
STATUS_REFERENCE = "reference"
STATUS_UNWIRED = "not_wired"
STATUS_FAILED = "failed"

# 对比图的排版尺寸（像素）。格子够小才能一屏看全，够大才能看清褶皱。
CELL_W = 320
CELL_H = 480
LABEL_H = 34
GAP = 10
COLS_DEFAULT = 4


@dataclass(slots=True)
class Candidate:
    """一个待比较的候选。id 用于产物命名与表格展示，必须唯一。"""

    id: str
    kind: str
    path: Path
    note: str = ""


@dataclass(slots=True)
class Outcome:
    """单个候选的处理结果。失败也是结果，必须能被记录下来。"""

    candidate: Candidate
    status: str
    output: Path | None = None
    elapsed_s: float = 0.0
    change: float | None = None
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.candidate.id,
            "kind": self.candidate.kind,
            "source": str(self.candidate.path),
            "status": self.status,
            "output": str(self.output) if self.output else None,
            "elapsed_s": round(self.elapsed_s, 1),
            "change": None if self.change is None else round(self.change, 2),
            "note": self.note,
        }


@dataclass(slots=True)
class CompareReport:
    person: Path
    out_dir: Path
    outcomes: list[Outcome] = field(default_factory=list)
    sheet: Path | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def ok_count(self) -> int:
        return sum(1 for o in self.outcomes if o.status in (STATUS_OK, STATUS_REFERENCE))

    def as_dict(self) -> dict[str, object]:
        return {
            "person": str(self.person),
            "out_dir": str(self.out_dir),
            "sheet": str(self.sheet) if self.sheet else None,
            "ok_count": self.ok_count,
            "total": len(self.outcomes),
            "outcomes": [o.as_dict() for o in self.outcomes],
            "notes": self.notes,
        }

    def table(self) -> str:
        """给终端看的对齐表格。中文列宽按显示宽度算，避免错位。"""
        rows = [("id", "kind", "status", "耗时", "差异", "说明")]
        for o in self.outcomes:
            rows.append(
                (
                    o.candidate.id,
                    o.candidate.kind,
                    o.status,
                    f"{o.elapsed_s:.0f}s",
                    "-" if o.change is None else f"{o.change:.1f}",
                    o.note[:40],
                )
            )
        widths = []
        for col in range(len(rows[0])):
            widths.append(max(_width(r[col]) for r in rows))
        lines = []
        for i, row in enumerate(rows):
            lines.append("  ".join(_pad(c, w) for c, w in zip(row, widths, strict=True)).rstrip())
            if i == 0:
                lines.append("  ".join("-" * w for w in widths))
        return "\n".join(lines)


def _width(text: str) -> int:
    """终端显示宽度：CJK 占两列。"""
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _width(text))


def infer_kind(path: Path) -> str:
    """按文件名前缀判断候选类型；无匹配前缀时按衣服处理。"""
    stem = path.stem.lower()
    head = stem.replace("-", "_").split("_", 1)[0]
    return PREFIX_TO_KIND.get(head, KIND_GARMENT)


def infer_category(path: Path) -> str | None:
    """按文件名里的单词猜试穿类别；猜不到返回 None（不硬猜）。"""
    for token in re.split(r"[^a-z0-9]+", path.stem.lower()):
        if token in CATEGORY_KEYWORDS:
            return CATEGORY_KEYWORDS[token]
    return None


def resolve_category(candidate: Candidate, category: str) -> tuple[str, str]:
    """把批次级别的 category 落实成这个候选的类别，并给出可展示的说明。

    `auto` 时逐候选判断；判断不出就用默认值，但说明里必须写清楚是默认值，
    否则用户会以为「你识别成了上衣」，其实是「我没识别出来」。
    """
    if category != CATEGORY_AUTO:
        return category, f"category={category}"
    guessed = infer_category(candidate.path)
    if guessed is not None:
        return guessed, f"category={guessed}（按文件名）"
    return CATEGORY_DEFAULT, f"category={CATEGORY_DEFAULT}（未识别，按默认）"


def discover(paths: list[Path]) -> list[Candidate]:
    """展开文件/目录为候选列表。

    目录按文件名排序（保证两次运行顺序一致，对比图可复现），
    同名候选自动加序号后缀，避免产物互相覆盖。
    """
    found: list[Path] = []
    for p in paths:
        if p.is_dir():
            found.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES))
        elif p.suffix.lower() in IMAGE_SUFFIXES:
            found.append(p)
        else:
            raise ValueError(f"不是图片或目录：{p}")

    used: dict[str, int] = {}
    out: list[Candidate] = []
    for f in found:
        base = f.stem
        count = used.get(base, 0)
        used[base] = count + 1
        cid = base if count == 0 else f"{base}_{count + 1}"
        out.append(Candidate(id=cid, kind=infer_kind(f), path=f))
    return out


def _mean_abs_diff(a: Path, b: Path) -> float | None:
    """与原图的平均像素差（0-255）。只用于提示「有没有变化」，不是质量分。"""
    try:
        from PIL import Image, ImageChops, ImageStat
    except ImportError:
        return None
    try:
        with Image.open(a) as ia, Image.open(b) as ib:
            left = ia.convert("RGB")
            right = ib.convert("RGB").resize(left.size)
            # 缩到 64×64 再取均值：box 滤波保均值，同时躲开对大图逐像素求和。
            # 用 ImageStat 而不是 getdata()，后者在 Pillow 14 会被移除。
            diff = ImageChops.difference(left, right).convert("L").resize((64, 64))
            stat = ImageStat.Stat(diff)
    except (OSError, ValueError):
        return None
    return float(stat.mean[0])


def _dispatch(
    candidate: Candidate,
    person: Path,
    out_dir: Path,
    *,
    vton: TryOnModel | None,
    hair: HairModel | None,
    category: str,
    hair_color: str | None,
) -> Outcome:
    """把一个候选交给对应后端。所有可预期失败都转成状态，不向上抛。"""
    started = time.monotonic()

    def done(status: str, output: Path | None, note: str = "") -> Outcome:
        return Outcome(
            candidate=candidate,
            status=status,
            output=output,
            elapsed_s=time.monotonic() - started,
            change=_mean_abs_diff(person, output) if output and output.exists() else None,
            note=note,
        )

    if candidate.kind == KIND_WORN:
        # 参照图：不渲染，只登记。它的价值是和「我穿上的样子」并排看。
        return done(STATUS_REFERENCE, candidate.path, "参照上身图，不参与试穿")

    category_value, category_note = resolve_category(candidate, category)
    try:
        if candidate.kind == KIND_HAIR:
            if hair is None:
                return done(STATUS_UNWIRED, None, "未配置发型模型")
            produced = hair.transfer(person, candidate.path, color=hair_color)
            status, note = STATUS_OK, "发型迁移"
        else:
            if vton is None:
                return done(STATUS_UNWIRED, None, "未配置试穿模型")
            produced = vton.apply(
                person, candidate.path, candidate.id, category=category_value
            ).image_path
            status, note = STATUS_OK, f"试穿 {category_note}"
    except ModelNotWired as exc:
        return done(STATUS_UNWIRED, None, str(exc).splitlines()[0])
    except Exception as exc:  # noqa: BLE001 - 单个候选失败不该中断整轮比较
        return done(STATUS_FAILED, None, f"{type(exc).__name__}: {exc}"[:120])

    # 结果图统一收进对比目录，路径稳定，报告里不出现后端各自的落盘位置
    target = out_dir / f"{candidate.id}.png"
    try:
        target.write_bytes(Path(produced).read_bytes())
    except OSError as exc:
        return done(STATUS_FAILED, None, f"产物复制失败：{exc}")
    return done(status, target, note)


def _fit(img, box_w: int, box_h: int):
    """等比缩放并居中裁切到目标框（cover 语义），保证每格人像高度一致。"""
    from PIL import Image

    w, h = img.size
    scale = max(box_w / w, box_h / h)
    resized = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    left = (resized.width - box_w) // 2
    top = (resized.height - box_h) // 2
    return resized.crop((left, top, left + box_w, top + box_h))


_STATUS_COLOR = {
    STATUS_OK: (90, 200, 140),
    STATUS_REFERENCE: (110, 160, 235),
    STATUS_UNWIRED: (230, 180, 90),
    STATUS_FAILED: (230, 110, 110),
}


def contact_sheet(
    report: CompareReport,
    person: Path,
    out_path: Path,
    cols: int = COLS_DEFAULT,
) -> Path | None:
    """把原图与所有候选拼成一张对比图。

    标签只用 ASCII（id / kind / status）：对比图不依赖任何中文字体文件，
    在服务器上跑也不会变成方框。
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        report.notes.append("未安装 Pillow，跳过对比图；产物仍在各自路径下")
        return None

    cells: list[tuple[str, str, Path | None, tuple[int, int, int]]] = [
        ("original", "baseline", person, (150, 150, 150))
    ]
    for o in report.outcomes:
        color = _STATUS_COLOR.get(o.status, (150, 150, 150))
        cells.append((o.candidate.id, f"{o.candidate.kind}/{o.status}", o.output, color))

    cols = max(1, min(cols, len(cells)))
    rows = (len(cells) + cols - 1) // cols
    width = cols * CELL_W + (cols + 1) * GAP
    height = rows * (CELL_H + LABEL_H) + (rows + 1) * GAP

    sheet = Image.new("RGB", (width, height), (18, 22, 28))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:  # Pillow < 10.1 不支持 size 参数
        font = ImageFont.load_default()

    for i, (label, sub, path, color) in enumerate(cells):
        cx = GAP + (i % cols) * (CELL_W + GAP)
        cy = GAP + (i // cols) * (CELL_H + LABEL_H + GAP)

        draw.rectangle([cx, cy, cx + CELL_W, cy + CELL_H + LABEL_H], outline=color, width=2)
        if path and path.exists():
            try:
                with Image.open(path) as img:
                    sheet.paste(_fit(img.convert("RGB"), CELL_W - 4, CELL_H - 4), (cx + 2, cy + 2))
            except (OSError, ValueError):
                draw.text((cx + 10, cy + CELL_H // 2), "unreadable", fill=(200, 120, 120), font=font)
        else:
            draw.text(
                (cx + 10, cy + CELL_H // 2),
                "no output",
                fill=(150, 150, 160),
                font=font,
            )
        draw.text((cx + 8, cy + CELL_H + 8), label[:34], fill=(230, 236, 244), font=font)
        draw.text((cx + 8, cy + CELL_H + 21), sub[:40], fill=color, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, "PNG")
    return out_path


def run(
    person: Path,
    candidates: list[Candidate],
    out_dir: Path,
    *,
    vton: TryOnModel | None = None,
    hair: HairModel | None = None,
    category: str = CATEGORY_AUTO,
    hair_color: str | None = None,
    cols: int = COLS_DEFAULT,
) -> CompareReport:
    """逐个渲染候选并生成对比图。

    顺序即对比顺序：不做任何自动排序，因为「哪件更好看」没有可计算的答案。
    """
    if category not in (*CATEGORIES, CATEGORY_AUTO):
        raise ValueError(f"未知 category：{category}（可选 {(*CATEGORIES, CATEGORY_AUTO)}）")

    out_dir.mkdir(parents=True, exist_ok=True)
    report = CompareReport(person=person, out_dir=out_dir)

    for cand in candidates:
        report.outcomes.append(
            _dispatch(
                cand,
                person,
                out_dir,
                vton=vton,
                hair=hair,
                category=category,
                hair_color=hair_color,
            )
        )

    report.sheet = contact_sheet(report, person, out_dir / "compare.png", cols=cols)
    (out_dir / "compare.json").write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


__all__ = [
    "CATEGORIES",
    "CATEGORY_AUTO",
    "CATEGORY_DEFAULT",
    "CELL_H",
    "CELL_W",
    "COLS_DEFAULT",
    "KINDS",
    "KIND_GARMENT",
    "KIND_HAIR",
    "KIND_WORN",
    "Candidate",
    "CompareReport",
    "Outcome",
    "contact_sheet",
    "discover",
    "infer_category",
    "infer_kind",
    "resolve_category",
    "run",
]
