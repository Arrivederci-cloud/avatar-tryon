"""环境自检 CLI：一条命令看清「哪一档能跑、缺什么、怎么补」。

    python scripts/doctor.py
    python scripts/doctor.py --offline    # 不联网，只看本地
    python scripts/doctor.py --json

退出码：基础依赖（proxy 档所需）缺失时为 1，否则 0。
real 档的缺失只报告、不判失败——它本来就依赖显卡与账号权限。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avatar_tryon.doctor import render, run_checks, to_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="环境自检")
    p.add_argument("--offline", action="store_true", help="跳过联网探测")
    p.add_argument("--json", action="store_true", help="输出 JSON")
    args = p.parse_args(argv)

    # Windows 控制台默认 GBK，不显式改会让中文全部变成乱码
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    report = run_checks(ROOT, offline=args.offline)
    print("环境自检（只读，不产生费用）\n")
    print(to_json(report) if args.json else render(report))

    # 只有 proxy 档的阻塞项算失败：real 档缺显卡/权重是常态
    hard = [c for c in report.blocking if "proxy" in c.tier]
    if hard:
        print(f"\n{len(hard)} 项阻塞 proxy 档：", file=sys.stderr)
        for c in hard:
            print(f"  - {c.name}: {c.detail}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
