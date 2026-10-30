"""基线纹理补全：最近源初始化 + 局部扩散，不需要任何生成模型。

作用与边界：
- 它把 UV 图里「没有任何机位看见」的区域用**周围的真实颜色**铺开，而不是全图平均
  （平均会把蓝衬衫糊成灰蓝）；
- 它不会生成新纹理，只是把已有颜色按最合理的方式外推，细节会丢；
- 上线时这一层换成可控重绘模型，接口不变，所以它同时是降级实现与对照基线。

算法（两段，都在待补区内操作，已知像素一个字节都不动）：

    1. 最近源初始化（Voronoi）：每个待补 texel 取二维上最近的已知 texel 颜色。
       这一步保证初始颜色来自**附近**，不会出现「整行被抹成一条灰带」。
       历史实现是轴向（行/列）最近邻，整行缺色时会把整行抹平，
       实测在手臂与腿的图集块上形成成片灰条。
    2. 局部扩散（Jacobi 迭代求解拉普拉斯方程，已知区作边界条件）：
       把 Voronoi 的块状边界磨成平滑过渡，视觉上才不像贴了补丁。
       只迭代 80 次即可，256² 图上约 30ms。

纯 numpy 实现（可选依赖）；没有 numpy 时退回 Pillow 金字塔填充。
"""

from __future__ import annotations

from pathlib import Path

from .base import EditModel

BLUR_RADIUS = 2
MIN_LEVEL = 8
DIFFUSE_ITERS = 80


class BaselineInpaint(EditModel):
    """实现 EditModel 协议：inpaint(image, mask, prompt) -> Path。"""

    name = "baseline-inpaint"
    license = "Apache-2.0（本仓库原创）"

    def __init__(self, blur_radius: int = BLUR_RADIUS, iterations: int = DIFFUSE_ITERS) -> None:
        self.blur_radius = blur_radius
        self.iterations = iterations

    def inpaint(self, image: Path, mask: Path, prompt: str) -> Path:
        from PIL import Image

        src = Image.open(image).convert("RGB")
        m = Image.open(mask).convert("L")
        if m.size != src.size:
            m = m.resize(src.size, Image.NEAREST)
        # 掩码约定：白(>=128) = 待补，黑 = 保留
        known = m.point(lambda v: 0 if v >= 128 else 255)

        try:
            filled = _diffused_fill(src, known, self.blur_radius, self.iterations)
        except ImportError:
            filled = _pyramid_fill(src, known, MIN_LEVEL)

        out = image.parent / "inpaint" / f"{image.stem}_filled.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        filled.save(out, "PNG")
        return out


def _diffused_fill(rgb, known, blur_radius: int, iterations: int):
    import numpy as np
    from PIL import Image, ImageFilter

    arr = np.asarray(rgb, dtype=np.float32)
    kn = np.asarray(known, dtype=np.uint8) > 127  # True = 已知
    if not kn.any():
        return rgb
    hole = ~kn
    if not hole.any():
        return rgb

    out = _nearest_source(arr, kn)
    out = _diffuse(out, kn, iterations)

    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")
    if blur_radius > 0:
        soft = img.filter(ImageFilter.GaussianBlur(blur_radius))
        keep = Image.fromarray(np.where(hole, 0, 255).astype(np.uint8), "L")
        # keep=255 在已知区：composite 在 mask=255 处取第一个参数，
        # 即已知区保留原图、待补区取柔化后的填充
        img = Image.composite(img, soft, keep)
    return img


def _nearest_source(arr, kn):
    """每个待补像素取最近的已知像素颜色（Voronoi）。cv2 缺失时用逐圈膨胀近似。"""
    import numpy as np

    hole = ~kn
    try:
        import cv2

        labels = cv2.distanceTransformWithLabels(
            hole.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )[1]
        # DIST_LABEL_PIXEL：已知像素自身 label=0，其余 1..N 按行优先对应零像素
        zeros = np.argwhere(~hole)
        out = arr.copy()
        idx = labels[hole] - 1
        src_px = zeros[idx]
        out[hole] = arr[src_px[:, 0], src_px[:, 1]]
        return out
    except ImportError:
        pass

    # 退路：逐圈向 4 邻域扩散，已填的像素不再被覆盖（近似最近邻）
    out = arr.copy()
    known = kn.copy()
    for _ in range(max(arr.shape[:2]) + 1):
        if known.all():
            break
        grown = known.copy()
        acc = out.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted = np.roll(np.roll(out, dy, axis=0), dx, axis=1)
            mask = np.roll(np.roll(known, dy, axis=0), dx, axis=1)
            take = (~grown) & mask
            acc[take] = shifted[take]
            grown |= take
        out, known = acc, grown
    return out


def _diffuse(arr, kn, iterations: int):
    """待补区做 Jacobi 迭代，已知像素作边界条件（保证已知区零改动）。"""
    import numpy as np

    if iterations <= 0:
        return arr
    hole = ~kn
    cur = arr.copy()
    for _ in range(iterations):
        # 边缘复制后再取四邻域：避免 np.roll 的环绕把图集对边连起来
        padded = np.pad(cur, ((1, 1), (1, 1), (0, 0)), mode="edge")
        nb = (
            padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
        )
        cur = np.where(hole[:, :, None], nb / 4.0, cur)
    return cur


def _pyramid_fill(rgb, known, min_level: int):
    """无 numpy 时的退路：层次化加权填充（偏模糊，但保证可用）。"""
    from PIL import Image

    levels: list[tuple[object, object]] = [(rgb, known)]
    cur_rgb, cur_kn = rgb, known
    while min(cur_rgb.size) > min_level:
        size = (max(1, cur_rgb.width // 2), max(1, cur_rgb.height // 2))
        cur_rgb = cur_rgb.resize(size, Image.BOX)
        cur_kn = cur_kn.resize(size, Image.BOX)
        levels.append((cur_rgb, cur_kn))

    coarse_rgb, coarse_kn = levels[-1]
    mean = _mean_known(coarse_rgb, coarse_kn)
    filled = Image.composite(coarse_rgb, Image.new("RGB", coarse_rgb.size, mean), coarse_kn)

    for rgb_l, kn_l in reversed(levels[:-1]):
        up = filled.resize(rgb_l.size, Image.BILINEAR)
        filled = Image.composite(rgb_l, up, kn_l)
    return filled


def _mean_known(rgb, known) -> tuple[int, int, int]:
    from PIL import Image, ImageStat

    masked = Image.composite(rgb, Image.new("RGB", rgb.size, (0, 0, 0)), known)
    sums = ImageStat.Stat(masked).sum  # [r, g, b] 各通道累加
    count = sum(ImageStat.Stat(known).sum) / 255.0  # 已知像素数
    if count <= 0:
        return (128, 128, 128)
    return tuple(int(min(255, round(c / count))) for c in sums)  # type: ignore[return-value]


__all__ = ["BaselineInpaint"]
