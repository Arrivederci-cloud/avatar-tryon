"""环境自检：把所有「外部门槛」一次查清，而不是靠翻文档猜。

为什么需要它：本项目的失败点几乎都在环境而非代码——Python 版本、显卡、
权重、账号是否开通某个模型、密钥是否存在。这些东西散落在 README、
docs/ 与上游仓库之间，报错时又只给一句 "Model not exist"。doctor 把它们
收敛成一张表：**每一条都给出「现状 / 影响哪一档 / 怎么补」**。

三条纪律：
1. 不产生费用——只做模型存在性探测，绝不发起生成任务；
2. 不回显密钥——只报是否配置与长度；
3. 只读——不写权重、不改配置。

探测结果需要解释，不能只看状态码。实测（2026-10-30）同一句
"Model not exist" 会配两种状态码：

    404 + Model not exist  → 模型名是对的，但当前账号/工作空间未开通
    400 + Model not exist  → 模型名不存在（上游改名或地域不同）

这个区别决定了「去控制台开通」还是「去改配置」，所以医生必须区分。
区别的判据来自对照实验：用一个必然不存在的模型名打同一端点得 400，
真实的 aitryon 得 404。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/api/v1"
FASHN_BASE = "https://api.fashn.ai"

# LHM 的依赖把 Python 上限钉死在 3.12：torch==2.3.0、numpy==1.23.0、chumpy、
# decord==0.6.0、xatlas 都没有 cp313/cp314 的轮子。
LHM_MAX_PYTHON = (3, 12)

# 用于判定「这一档能不能跑」的最小依赖
PROXY_DEPS = (("PIL", "Pillow"), ("numpy", "numpy"), ("cv2", "opencv-python"))

OK = "ok"
WARN = "warn"
BLOCK = "block"

_MARK = {OK: "[ok]  ", WARN: "[warn]", BLOCK: "[block]"}


@dataclass
class Check:
    """一条检查结果。tier 说明它影响哪个引擎档位。"""

    name: str
    status: str
    detail: str
    action: str = ""
    tier: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> None:
        self.checks.append(check)

    @property
    def blocking(self) -> list[Check]:
        return [c for c in self.checks if c.status == BLOCK]

    def by_tier(self) -> dict[str, str]:
        """每个档位最差的状态：block > warn > ok。

        注意不能在 out 里用默认值兜底：全 ok 的档位也需要出现，
        否则汇总会凭空少一行，看起来像「这一档没检查过」。
        """
        order = {OK: 0, WARN: 1, BLOCK: 2}
        out: dict[str, str] = {}
        for c in self.checks:
            if not c.tier:
                continue
            for tier in c.tier.split(","):
                tier = tier.strip()
                if not tier:
                    continue
                current = out.get(tier)
                if current is None or order[c.status] > order[current]:
                    out[tier] = c.status
        return out


def classify_model_probe(status: int, body: str) -> tuple[str, str]:
    """把一次模型探测的响应解释成（结论, 人话说明）。

    结论取值：available / not_enabled / unknown_name / inconclusive。
    判据见模块头注释的对照实验，别改成只看状态码——只看 404 会把
    「模型名写错」误判成「没开通」。
    """
    text = (body or "").lower()
    model_missing = "model not exist" in text or "model_not_exist" in text

    if status == 200:
        return "available", "模型可用"
    if model_missing and status == 404:
        return "not_enabled", "模型名有效，但当前账号/工作空间未开通（去控制台开通）"
    if model_missing:
        return "unknown_name", "模型名不存在：上游可能改名或该地域不提供"
    # 走到这里说明模型名已被解析，报错来自参数校验——模型本身是通的
    if 400 <= status < 500:
        return "available", "模型名被接受（参数校验报错，属正常探测噪声）"
    return "inconclusive", f"无法判定（HTTP {status}）"


def check_python(version_info: tuple[int, ...] | None = None) -> Check:
    v = version_info or sys.version_info[:3]
    text = ".".join(str(x) for x in v[:3])
    if v[:2] <= LHM_MAX_PYTHON:
        return Check(
            "Python 版本", OK,
            f"{text}（在 LHM 可用区间 ≤{'.'.join(map(str, LHM_MAX_PYTHON))} 内）",
            tier="real",
        )
    return Check(
        "Python 版本", BLOCK,
        f"{text} 超出 LHM 可用区间 ≤{'.'.join(map(str, LHM_MAX_PYTHON))}；"
        f"LHM 钉住 torch==2.3.0 / numpy==1.23.0 / chumpy / decord / xatlas，均无 cp313+ 轮子",
        action=(
            "real 档要用独立环境："
            "py -3.12 -m venv .venv-lhm && .venv-lhm\\Scripts\\activate && pip install -r lhm-req.txt"
        ),
        tier="real",
    )


def check_module(modname: str, label: str, tier: str) -> Check:
    try:
        mod = __import__(modname)
    except ImportError:
        return Check(
            f"依赖 {label}", WARN, "未安装",
            action=f"pip install {label}", tier=tier,
        )
    version = getattr(mod, "__version__", "?")
    return Check(f"依赖 {label}", OK, f"{version}", tier=tier)


def _load_dotenv(root: Path) -> None:
    """复用 config.load_dotenv 的极简语义，但不引入包内依赖，保持脚本可独立运行。"""
    env_file = root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def check_key(name: str, tier: str, action: str) -> Check:
    value = os.environ.get(name, "")
    if not value:
        return Check(f"密钥 {name}", WARN, "未配置", action=action, tier=tier)
    # 只报长度与前缀族，绝不回显
    return Check(f"密钥 {name}", OK, f"已配置（{len(value)} 字符）", tier=tier)


def check_gpu() -> Check:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return Check(
            "显卡", WARN, "nvidia-smi 不在 PATH（无法判断，不代表没有显卡）",
            action="nvidia-smi 观察；无独显时只能用 mock/proxy 档", tier="real",
        )
    try:
        out = subprocess.run(  # noqa: S603
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return Check("显卡", WARN, f"nvidia-smi 调用失败：{type(exc).__name__}", tier="real")
    line = (out.stdout or "").strip().splitlines()
    if not line:
        return Check("显卡", WARN, "未检测到 GPU", tier="real")
    return Check("显卡", OK, line[0].strip(), tier="real")


def check_weights(weights_dir: Path) -> list[Check]:
    """权重是否就位。MANIFEST.json 由 scripts/fetch_weights.py 写入，用来钉住版本。"""
    checks: list[Check] = []
    for name, tier in (("lhm", "real"), ("mhr", "real"), ("fashn_vton", "real")):
        d = weights_dir / name
        if (d / "MANIFEST.json").exists():
            checks.append(Check(f"权重 {name}", OK, f"{d}", tier=tier))
        else:
            checks.append(
                Check(
                    f"权重 {name}", WARN, "未下载",
                    action=f"python scripts/fetch_weights.py --model {name}", tier=tier,
                )
            )
    return checks


def probe_model(url: str, payload: dict, api_key: str, async_hdr: bool = True) -> tuple[int, str]:
    """发一次只读探测：不满足参数就不会创建任务，因此不产生费用。"""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if async_hdr:
        headers["X-DashScope-Async"] = "enable"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, f'{{"error": "{type(exc).__name__}"}}'


def check_dashscope_models(api_key: str) -> list[Check]:
    """探测两个模型族：试穿（aitryon）与纹理编辑（wanx）。

    请求体故意不带图片 URL：既省流量，也保证不会真的创建任务。
    """
    checks: list[Check] = []
    probes = (
        (
            "试穿模型 aitryon",
            f"{DASHSCOPE_BASE}/services/aigc/image2image/image-synthesis/",
            {"model": "aitryon", "input": {}},
            "去 Model Studio 控制台开通「AI试衣」，或改用 FASHN（--vton-backend api + FASHN_API_KEY）",
            "real",
        ),
        (
            "纹理编辑 wanx2.1-imageedit",
            f"{DASHSCOPE_BASE}/services/aigc/image2image/image-synthesis/",
            {"model": "wanx2.1-imageedit", "input": {}},
            "开通后 --edit-backend api 可用",
            "proxy,real",
        ),
    )
    for label, url, payload, action, tier in probes:
        status, body = probe_model(url, payload, api_key)
        if status == 0:
            checks.append(Check(label, WARN, f"端点不可达（{body[:60]}）",
                                action="检查网络或代理", tier=tier))
            continue
        verdict, why = classify_model_probe(status, body)
        if verdict == "available":
            checks.append(Check(label, OK, why, tier=tier))
        elif verdict == "not_enabled":
            checks.append(Check(label, WARN, why, action=action, tier=tier))
        else:
            checks.append(Check(label, WARN, why, action=action, tier=tier))
    return checks


def check_fashn_reachable() -> Check:
    req = urllib.request.Request(FASHN_BASE, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            return Check("FASHN 端点", OK, f"可达（HTTP {resp.status}）", tier="real")
    except urllib.error.HTTPError as exc:
        # 4xx 也算可达：说明域名与路径对，只是没带鉴权
        return Check("FASHN 端点", OK, f"可达（HTTP {exc.code}）", tier="real")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return Check("FASHN 端点", WARN, f"不可达（{type(exc).__name__}）",
                     action="检查网络或代理", tier="real")


def run_checks(root: Path, offline: bool = False) -> Report:
    _load_dotenv(root)
    report = Report()

    report.add(check_python())
    for modname, label in PROXY_DEPS:
        report.add(check_module(modname, label, tier="proxy"))
    report.add(check_module("torch", "torch", tier="real"))
    report.add(check_gpu())
    for c in check_weights(Path(os.environ.get("AVATAR_WEIGHTS_DIR", root / "weights"))):
        report.add(c)

    dashscope = os.environ.get("DASHSCOPE_API_KEY", "")
    report.add(check_key(
        "DASHSCOPE_API_KEY", "proxy,real",
        "在 .env 填 DASHSCOPE_API_KEY=<key>（Model Studio 控制台创建）",
    ))
    report.add(check_key(
        "FASHN_API_KEY", "real",
        "在 .env 填 FASHN_API_KEY=<key>；或用 --vton-backend local 走本地权重",
    ))

    if offline:
        report.add(Check("联网探测", WARN, "已跳过（--offline）", tier=""))
        return report

    if dashscope:
        for c in check_dashscope_models(dashscope):
            report.add(c)
    else:
        report.add(Check("联网探测", WARN, "无 DASHSCOPE_API_KEY，跳过模型探测", tier=""))
    report.add(check_fashn_reachable())
    return report


def _display_width(text: str) -> int:
    """终端显示宽度：中日韩全角字符占两列，len() 会把列对齐搞错。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _display_width(text))


def render(report: Report) -> str:
    width = max((_display_width(c.name) for c in report.checks), default=0)
    lines = []
    for c in report.checks:
        lines.append(f"{_MARK[c.status]} {_pad(c.name, width)}  {c.detail}")
        if c.action and c.status != OK:
            lines.append(f"        {' ' * width}  -> {c.action}")
    tiers = report.by_tier()
    if tiers:
        lines.append("")
        for tier in ("mock", "proxy", "real"):
            if tier in tiers:
                lines.append(f"档位 {tier:5s}: {tiers[tier]}")
    return "\n".join(lines)


def to_json(report: Report) -> str:
    return json.dumps(
        {
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail,
                 "action": c.action, "tier": c.tier}
                for c in report.checks
            ],
            "tiers": report.by_tier(),
        },
        ensure_ascii=False,
        indent=2,
    )


__all__ = [
    "BLOCK",
    "Check",
    "OK",
    "Report",
    "WARN",
    "check_python",
    "classify_model_probe",
    "render",
    "run_checks",
    "to_json",
]
