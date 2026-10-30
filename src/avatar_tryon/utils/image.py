"""最小图像头解析：只读尺寸，不依赖 Pillow。"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path


def write_blank_png(path: Path, width: int = 1024, height: int = 1024) -> Path:
    """生成一张纯白 PNG。仅用于 mock 模式与测试，避免依赖 Pillow。"""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\xff" * (width * 3) for _ in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return path


def write_gray_png(path: Path, width: int, height: int, pixels) -> Path:
    """写 8-bit 灰度 PNG。pixels 为长度 width*height 的字节序列，逐行自上而下。

    仅用于 UV 观测掩码与测试，避免依赖 Pillow。
    """

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    buf = bytes(pixels)
    if len(buf) != width * height:
        raise ValueError(f"像素数量不符：期望 {width * height}，实际 {len(buf)}")
    raw = b"".join(b"\x00" + buf[y * width : (y + 1) * width] for y in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return path


def image_size(path: Path) -> tuple[int, int]:
    """返回 (width, height)。支持 PNG 与 JPEG。"""
    with path.open("rb") as f:
        head = f.read(26)

        # PNG: 8 字节签名 + IHDR
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            w, h = struct.unpack(">II", head[16:24])
            return int(w), int(h)

        # JPEG: 扫描 SOF 段
        if head.startswith(b"\xff\xd8"):
            f.seek(2)
            while True:
                byte = f.read(1)
                if not byte:
                    break
                if byte != b"\xff":
                    continue
                marker = f.read(1)
                if not marker:
                    break
                if marker[0] in (0xC0, 0xC1, 0xC2, 0xC3):
                    _ = f.read(3)
                    h, w = struct.unpack(">HH", f.read(4))
                    return int(w), int(h)
                seg_len = f.read(2)
                if len(seg_len) < 2:
                    break
                f.seek(int.from_bytes(seg_len, "big") - 2, 1)

    raise ValueError(f"无法识别图像尺寸（仅支持 PNG / JPEG）：{path}")
