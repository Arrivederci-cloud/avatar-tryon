"""下载 MediaPipe 姿态模型（支持断点续传与重试）。

这台机器走代理，单连接大约 6MB 就会被掐断，所以必须按 Range 分段续传。
产物落 weights/（已在 .gitignore），只需跑一次。
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "weights" / "mediapipe" / "pose_landmarker_full.task"
EXPECTED = 9_398_198  # 上游 Content-Length
MAX_TRIES = 20


def _size() -> int:
    return DEST.stat().st_size if DEST.exists() else 0


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if _size() == EXPECTED:
        print(f"已完整：{DEST}（{EXPECTED / 1e6:.1f} MB）")
        return 0
    if _size() > EXPECTED:
        DEST.unlink()

    for attempt in range(1, MAX_TRIES + 1):
        start = _size()
        if start == EXPECTED:
            break
        req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-"})
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                mode = "ab" if resp.status == 206 else "wb"
                if mode == "wb":
                    start = 0
                with DEST.open(mode) as f:
                    while True:
                        chunk = resp.read(512 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
        except Exception as exc:  # 代理会主动断连，属于预期
            print(f"  第 {attempt} 次中断于 {_size() / 1e6:.1f} MB（{type(exc).__name__}），续传")
            time.sleep(1.0)
            continue
        print(f"  第 {attempt} 次：{_size() / 1e6:.1f} MB")
        if _size() == EXPECTED:
            break

    got = _size()
    if got != EXPECTED:
        print(f"未完成：{got / 1e6:.1f} / {EXPECTED / 1e6:.1f} MB")
        return 1
    print(f"完成：{DEST}（{got / 1e6:.1f} MB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
