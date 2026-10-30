"""人物抠图：GrabCut，无需下载任何模型权重。

为什么需要它：三视图投影要把「人物像素」采进 UV。用「与背景色距离」这种
颜色启发式在渐变背景上会双重失效——既估不准背景色，也定不准人物包围盒
（实测边界像素 140→195 的渐变即触发）。GrabCut 用图割 + 颜色模型做前景分离，
对渐变背景稳健得多，且 OpenCV 一个 pip 包即可，不依赖任何需要联网下载的权重。

cv2 缺失时返回 None，调用方退回颜色启发式——保证零依赖环境仍能跑。
"""

from __future__ import annotations

MAX_SIDE = 512
MIN_REL_SPAN = 0.15


def available() -> bool:
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


def person_matte(src, rect=None, iterations: int = 5):
    """返回 (mask, bbox)；mask 为 'L' 模式（255=人物），bbox=(x0,y0,x1,y1)。

    失败或抠图结果不可信时返回 (None, None)，由调用方回退。

    可信度校验：GrabCut 在「rect 几乎盖住整图」时经常把整张图都判成前景
    （没有明确背景可供对比）。表现为前景 bbox 盖满 rect——这是失败信号，
    必须收缩后重试一次，否则背景会被当成人烘进 UV
    （实测整条手臂区被灰背景污染，正面纹理半边发灰）。
    """
    first, first_box = _attempt(src, rect, iterations)
    if first is None:
        return None, None
    if not _suspicious(first, src.size):
        return _validated(first)

    retry, _box2 = _attempt(src, None, iterations + 3)
    if retry is None or _suspicious(retry, src.size):
        return _validated(first)
    return _validated(retry)


def _validated(mask):
    bbox = mask.getbbox()
    if bbox is None:
        return None, None
    w, h = mask.size
    if (bbox[2] - bbox[0]) < w * MIN_REL_SPAN or (bbox[3] - bbox[1]) < h * MIN_REL_SPAN:
        return None, None
    return mask, bbox


def _suspicious(mask, size) -> bool:
    """前景 bbox 盖满整图视为抠图失败。

    用面积而不是「宽高同时 ≥97%」：失败时高度方向经常仍差几个点
    （实测 96.1%），用与条件会漏判。
    """
    bbox = mask.getbbox()
    if bbox is None:
        return True
    w, h = size
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return area >= w * h * 0.90


def _attempt(src, rect, iterations: int):
    """跑一次 GrabCut，返回 (mask, None) 或 (None, None)。"""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None, None

    from PIL import Image

    w, h = src.size
    scale = min(1.0, MAX_SIDE / max(w, h))
    small = src.resize((max(8, int(w * scale)), max(8, int(h * scale)))) if scale < 1.0 else src.copy()
    sw, sh = small.size

    if rect is None:
        box = (1, 1, sw - 2, sh - 2)
    else:
        box = (
            max(1, int(rect[0] * scale)),
            max(1, int(rect[1] * scale)),
            min(sw - 2, int(rect[2] * scale)),
            min(sh - 2, int(rect[3] * scale)),
        )
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            box = (1, 1, sw - 2, sh - 2)

    arr = np.asarray(small.convert("RGB"))[:, :, ::-1].copy()
    mask = np.zeros(arr.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(arr, mask, box, bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None, None

    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

    out = Image.fromarray(fg, "L")
    if out.size != src.size:
        out = out.resize(src.size, Image.NEAREST)
    return out, None


def erode(mask, radius: int = 1):
    """向内收缩掩码 radius 像素；cv2 缺失或失败时返回原掩码。

    用途：贴图采样时避开轮廓边缘的抗锯齿混色带——边缘像素混了背景色，
    直接采进 UV 会在模型边缘形成灰边。返回 None 表示不可用。
    """
    if mask is None or radius <= 0:
        return mask
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError:
        return mask
    arr = np.asarray(mask)
    if arr.ndim != 2:
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    out = cv2.erode(arr, kernel, iterations=1)
    return Image.fromarray(out, "L")


__all__ = ["available", "erode", "person_matte"]
