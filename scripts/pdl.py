"""分块断点下载器：为「大权重 + 国内网络」准备的最后手段。

为什么不用 huggingface_hub / modelscope 官方 SDK：实测（见
docs/environment-gates.md）HF 直连会被中间代理破坏、hf-mirror 会 429、
ModelScope SDK 并发被限速到 0.27 MB/s。单连接直取反而最快（约 2.5 MB/s），
所以这个工具默认单线程、分 8 MB 一片、断点续传；并发参数保留给带宽充足的场景。

用法：
    python scripts/pdl.py <url> <目标文件> [并发数]

示例（权重 1.94 GB，国内实测约 13 分钟）：
    python scripts/pdl.py "https://modelscope.cn/api/v1/models/fashn-ai/fashn-vton-1.5/repo?Revision=master&FilePath=model.safetensors" weights/fashn_vton/model.safetensors

支持断点：已存在的分块不重下（分块文件留在目标同名 .parts 目录里）。
下载完成后自动按序拼接，.parts 目录可手动删除。
"""

import concurrent.futures as cf
import sys
import threading
import time
import urllib.request
from pathlib import Path

CHUNK = 8 * 1024 * 1024  # 8 MB 一片，太大不利于失败重试，太小握手开销高


def total_size(url: str) -> int:
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "probe"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        cr = resp.headers.get("Content-Range", "")
        resp.read()
    return int(cr.rsplit("/", 1)[-1])


def fetch_range(url: str, start: int, end: int, dest: Path, retries: int = 3) -> None:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "probe"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
                data = resp.read()
            if len(data) != end - start + 1:
                raise OSError(f"长度不符 {len(data)} != {end - start + 1}")
            dest.write_bytes(data)
            return
        except Exception:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def main() -> int:
    url, out = sys.argv[1], Path(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 6

    size = total_size(url)
    parts = out.with_suffix(out.suffix + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    ranges = [(i, s, min(s + CHUNK - 1, size - 1)) for i, s in enumerate(range(0, size, CHUNK))]
    todo = [r for r in ranges if not (parts / f"{r[0]:05d}").exists()]
    print(f"总大小 {size / 1e9:.2f} GB，{len(ranges)} 片，待下 {len(todo)} 片，{workers} 线程", flush=True)

    t0 = time.monotonic()
    done = 0
    lock = threading.Lock()

    def work(item: tuple[int, int, int]) -> None:
        nonlocal done
        idx, start, end = item
        fetch_range(url, start, end, parts / f"{idx:05d}")
        with lock:
            done += 1
            if done % 10 == 0 or done == len(todo):
                dt = time.monotonic() - t0
                speed = done * CHUNK / 1e6 / max(dt, 1e-6)
                print(f"  {done}/{len(todo)} 片  {speed:.1f} MB/s", flush=True)

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, todo))

    # 按序拼接
    print("拼接……", flush=True)
    with out.open("wb") as fh:
        for idx, _, _ in ranges:
            fh.write((parts / f"{idx:05d}").read_bytes())
    print(f"完成：{out}  {out.stat().st_size / 1e9:.2f} GB  {time.monotonic() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
