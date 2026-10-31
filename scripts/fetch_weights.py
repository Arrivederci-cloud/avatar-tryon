"""权重拉取脚本。

三条纪律：
1. 权重永远不进入 git。它们放到 weights/，而 weights/ 已在 .gitignore 里。
2. 锁定 revision 与 sha256，避免上游静默更新导致结果漂移。
3. 只从官方渠道拉取。受限模型（SAM License 等）本脚本不代下载，
   需要使用者自己去官方页面接受条款。

国内网络提示（实测，见 docs/environment-gates.md）：
- HF 直连可能被中间代理破坏（返回缺 X-Repo-Commit 头），先设
  `HF_ENDPOINT=https://hf-mirror.com`；
- 单个大文件（GB 级）用 huggingface_hub 容易 429 或极慢，改用
  `python scripts/pdl.py "<ModelScope 直链>" <目标路径>` 分块断点下载，
  实测比 SDK 快一个数量级。

用法：
    python scripts/fetch_weights.py --model lhm
    python scripts/fetch_weights.py --model all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path("weights")

# revision 与 sha256 需要首次拉取后回填。留空时脚本会拉取并在结束时打印实际值。
CATALOG: dict[str, dict] = {
    "lhm": {
        "repo": "aigc3d/LHM",
        "revision": "main",
        "allow_patterns": ["*.json", "*.safetensors", "*.yaml"],
        "license": "Apache-2.0",
        "sha256": None,
    },
    "mhr": {
        "repo": "facebookresearch/MHR",
        "revision": "main",
        "allow_patterns": ["*.pt", "*.npz", "*.json"],
        "license": "Apache-2.0",
        "sha256": None,
    },
    "fashn_vton": {
        "repo": "fashn-AI/fashn-vton-1.5",
        "revision": "main",
        "allow_patterns": ["*.safetensors", "*.json"],
        "license": "Apache-2.0",
        "sha256": None,
    },
    "hairfastgan": {
        "repo": "AIRI-Institute/HairFastGAN",
        "revision": "main",
        "allow_patterns": ["*.pt", "*.pkl", "*.json"],
        "license": "MIT",
        "sha256": None,
    },
    "qwen_image_edit": {
        "repo": "Qwen/Qwen-Image-Edit",
        "revision": "main",
        "allow_patterns": ["*.safetensors", "*.json", "*.txt"],
        "license": "Apache-2.0",
        "sha256": None,
    },
    "unirig": {
        "repo": "VAST-AI/UniRig",
        "revision": "main",
        "allow_patterns": ["*.pt", "*.json"],
        "license": "MIT",
        "sha256": None,
    },
}

NOT_DOWNLOADED = {
    "sam3d_body": "SAM License (Meta)：需自行前往官方页面接受条款后下载，本脚本不代下",
    "magicman": "未见明确许可：默认保留所有权利，请勿下载后随仓库分发",
    "catvton": "CC BY-NC-SA 4.0：禁止商用，仅可用于本地研究",
}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def fetch(name: str) -> int:
    if name in NOT_DOWNLOADED:
        print(f"[skip] {name}: {NOT_DOWNLOADED[name]}", file=sys.stderr)
        return 2

    spec = CATALOG[name]
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("缺少依赖：pip install huggingface_hub", file=sys.stderr)
        return 2

    target = WEIGHTS / name
    print(f"[fetch] {name} <- {spec['repo']} @ {spec['revision']} ({spec['license']})")
    snapshot_download(
        repo_id=spec["repo"],
        revision=spec["revision"],
        local_dir=str(target),
        allow_patterns=spec["allow_patterns"],
    )

    manifest = {"repo": spec["repo"], "revision": spec["revision"], "license": spec["license"], "files": {}}
    for f in sorted(target.rglob("*")):
        if f.is_file():
            manifest["files"][str(f.relative_to(target))] = sha256_of(f)
    (target / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[ok] {name} -> {target}  校验清单已写入 MANIFEST.json")
    print("[提醒] 请把 MANIFEST.json 里的哈希回填到本脚本 CATALOG，锁定版本。")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="拉取模型权重")
    p.add_argument("--model", required=True, help="模型名或 all")
    p.add_argument("--dest", type=Path, default=WEIGHTS)
    args = p.parse_args(argv)

    global WEIGHTS
    WEIGHTS = args.dest

    names = list(CATALOG) + list(NOT_DOWNLOADED) if args.model == "all" else [args.model]
    rc = 0
    for n in names:
        rc |= fetch(n)
    return 0 if rc in (0, 2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
