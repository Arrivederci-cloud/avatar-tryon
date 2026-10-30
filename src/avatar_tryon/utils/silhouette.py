"""人物轮廓宽度剖面。

为什么需要它：把网格表面投影进照片时，如果只用「网格总宽度 -> 人物包围盒」
这一个线性映射，网格比真人宽的部位（典型是外展的手臂）就会落到人物轮廓之外的
背景上，抠图判定为背景 -> 这些 texel 被丢弃 -> UV 残留待补面积飙升。

实测这一条就是代理几何 39% 残留的主因：代理手臂末端到身体中线的距离是 23.8cm，
而真人垂手站立时该高度处的轮廓半宽只有 ~18cm，外沿 6cm 的环带全部落在背景里。

正确的约束不是「总宽度对得上」，而是逐高度对得上：
    网格在高度 y 处的横向范围  ->  照片在高度 y 处的横向范围
这是一个逐带仿射映射，既保证贴图不越界，又保留纵向比例。

只依赖抠图掩码，不需要姿态关键点。
"""

from __future__ import annotations

# 纵向分带数。96 带对 512px 高的图约等于 5px 一带，足够跟随腰线收放。
BANDS = 96

# 一带内前景像素少于该值时视为空带（噪点），由邻带补位
MIN_BAND_PIXELS = 2


class Profile:
    """逐带轮廓：lo/hi 为该带前景左右边界（像素，可为 None）。"""

    __slots__ = ("bands", "lo", "hi", "bbox")

    def __init__(self, bands: int, lo: list, hi: list, bbox) -> None:
        self.bands = bands
        self.lo = lo
        self.hi = hi
        self.bbox = bbox

    def span(self, band: int):
        """返回该带的 (lo, hi)；空带向上下邻带借位。"""
        if not self.lo:
            return None
        b = min(max(band, 0), self.bands - 1)
        for d in range(0, self.bands):
            for i in (b - d, b + d):
                if 0 <= i < self.bands and self.lo[i] is not None:
                    return self.lo[i], self.hi[i]
            if d > 8:
                break
        return None


def fill_gaps(lo, hi) -> None:
    """把空带用最近的已知带补齐（原地修改），使调用方可以 O(1) 直接索引。"""
    known = [i for i, v in enumerate(lo) if v is not None]
    if not known:
        return
    n = len(lo)
    for b in range(n):
        if lo[b] is not None:
            continue
        # 找最近的已知带（known 已排序，bisect 语义）
        j = 0
        while j < len(known) and known[j] < b:
            j += 1
        cand = []
        if j < len(known):
            cand.append(known[j])
        if j > 0:
            cand.append(known[j - 1])
        if not cand:
            continue
        src = min(cand, key=lambda i: abs(i - b))
        lo[b], hi[b] = lo[src], hi[src]


def width_profile(mask, bands: int = BANDS) -> Profile | None:
    """把 'L' 模式人物掩码压成宽度剖面。

    mask 为 None 或无前景时返回 None。坐标沿用 mask 自身像素坐标。
    """
    if mask is None:
        return None
    bbox = mask.getbbox()
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    span = y1 - y0
    if span < 8 or (x1 - x0) < 4:
        return None

    lo: list = [None] * bands
    hi: list = [None] * bands
    counts = [0] * bands

    rows = _rows(mask)
    if rows is None:
        return None
    for y, left, right in rows:
        if y < y0 or y >= y1:
            continue
        b = min(bands - 1, max(0, int((y - y0) * bands / span)))
        if lo[b] is None:
            lo[b], hi[b] = left, right
        else:
            lo[b] = min(lo[b], left)
            hi[b] = max(hi[b], right)
        counts[b] += 1

    for b in range(bands):
        if counts[b] < MIN_BAND_PIXELS or lo[b] is None or hi[b] - lo[b] < 2:
            lo[b] = hi[b] = None

    _smooth(lo, hi)
    if not any(v is not None for v in lo):
        return None
    fill_gaps(lo, hi)
    return Profile(bands, lo, hi, bbox)


def _rows(mask):
    """逐行返回 (y, 最左前景 x, 最右前景 x)。优先 numpy，缺失时用 PIL 扫描。"""
    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None:
        arr = np.asarray(mask)
        if arr.ndim != 2:
            return None
        cols = np.nonzero(arr > 127)
        if cols[0].size == 0:
            return None
        ys, xs = cols
        return _gather(np, ys, xs)

    w, h = mask.size
    px = mask.load()
    step = max(1, w // 256)
    out = []
    for y in range(h):
        left = right = None
        for x in range(0, w, step):
            if px[x, y] > 127:
                if left is None:
                    left = x
                right = x
        if left is not None:
            out.append((y, left, right))
    return out


def _gather(np, ys, xs):
    """按行聚合 min/max（向量化，避免 Python 逐像素循环）。"""
    order = np.argsort(ys, kind="stable")
    ys = ys[order]
    xs = xs[order]
    uniq, start = np.unique(ys, return_index=True)
    xmin = np.minimum.reduceat(xs, start)
    xmax = np.maximum.reduceat(xs, start)
    return [
        (int(y), int(a), int(b))
        for y, a, b in zip(uniq.tolist(), xmin.tolist(), xmax.tolist(), strict=False)
    ]


def _smooth(lo, hi, radius: int = 1) -> None:
    """滑动平均，抑制单带抖动（原地修改）。"""
    n = len(lo)
    ol, oh = list(lo), list(hi)
    for b in range(n):
        if ol[b] is None:
            continue
        acc_l = acc_h = 0.0
        w = 0
        for i in range(max(0, b - radius), min(n, b + radius + 1)):
            if ol[i] is None:
                continue
            acc_l += ol[i]
            acc_h += oh[i]
            w += 1
        if w:
            lo[b] = acc_l / w
            hi[b] = acc_h / w


def band_of(v: float, bands: int) -> int:
    """v∈[0,1]（0=头顶）-> 带索引。"""
    return min(bands - 1, max(0, int(v * bands)))


def mesh_ranges(verts, right, y_lo: float, y_span: float, bands: int):
    """网格沿 right 轴、按纵向带统计的 min/max 投影范围。

    返回两个长度 bands 的列表；某带没有顶点时为 None。
    """
    lo: list = [None] * bands
    hi: list = [None] * bands
    for x, y, z in verts:
        p = x * right[0] + z * right[2]
        v = 1.0 - (y - y_lo) / y_span  # 与烘焙时一致：0=头顶
        b = band_of(v, bands)
        if lo[b] is None:
            lo[b], hi[b] = p, p
        else:
            if p < lo[b]:
                lo[b] = p
            if p > hi[b]:
                hi[b] = p
    fill_gaps(lo, hi)
    return lo, hi


__all__ = [
    "BANDS",
    "Profile",
    "band_of",
    "fill_gaps",
    "mesh_ranges",
    "width_profile",
]
