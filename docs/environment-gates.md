# 环境门槛：为什么这台机器现在只能跑 proxy 档

代码是通的，卡住的是运行环境、账号权限与权重体积。这份文档把「哪一档能跑、
缺什么、为什么」写成可复核的实测记录，避免下一个人重复踩。

所有数字来自 2026-10-30 在本机（Windows / Python 3.14.5 / RTX 4060 Laptop 8 GB）
的实测。复核命令：`python scripts/doctor.py`。

## 一、运行时代差：LHM 的依赖上限是 Python 3.12

`real` 档的后端是 [LHM](https://github.com/aigc3d/LHM)。它的 `requirements.txt`
是一组强钉版本，其中几项在 Python 3.13/3.14 上**没有轮子**：

| 依赖 | LHM 钉的版本 | cp314 轮子 | 后果 |
|---|---|---|---|
| `torch` | `==2.3.0` | 只有 2.15.0 有 | 装得上 2.15 但不满足钉死版本 |
| `numpy` | `==1.23.0` | 无（1.x 全系不支持 3.14） | 源码编译必失败 |
| `chumpy` | 无钉版本 | 无 | 该包早已停止维护，`inspect.getargspec` 在新版本已移除 |
| `decord` | `==0.6.0` | 无 Windows 轮子 | 只能编译 |
| `xatlas` / `gsplat` | `0.0.9` / `1.4.0` | 无 | `gsplat` 需 CUDA toolkit + MSVC 现场编译 |

结论：**这不是网络问题，是运行时代差**。PyPI 上确实有 `torch-2.15.0-cp314`
（126.7 MB，本机实测 2.8 MB/s 下载无阻），但 LHM 要的是 2.3.0 那套 ABI，
换不过去。`real` 档必须用独立的 Python 3.12 环境：

```bash
py -3.12 -m venv .venv-lhm
.venv-lhm\Scripts\activate
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r <LHM 仓库>/requirements.txt     # 先装官方 requirements，再处理 gsplat
```

`gsplat` 与 `basicsr` 在 Windows 上需要 CUDA toolkit 与 MSVC 现场编译，这是整条
链路上最容易失败的一步。**预算按半天到一天算，不要在演示前夜做这件事。**

## 二、账号门槛：试穿模型存在，但没给这个账号开通

阿里云 Model Studio 的 AI 试衣家族是 `aitryon` / `aitryon-plus`（试衣）与
`aitryon-parsing-v1`（服饰分割）。用现有 key 探测，返回的是：

```
HTTP 404  {"code":"InvalidParameter","message":"Model not exist."}
```

单看这句话会误判成「模型名写错了」。做一次对照实验就能区分：

| 探测对象 | 状态码 | 文案 | 结论 |
|---|---|---|---|
| `aitryon`（真实模型） | **404** | Model not exist. | 名字有效，**账号未开通** |
| `aitryon-parsing-v1`（真实模型） | **404** | Model not exist. | 同上（同族，不同端点） |
| `totally-bogus-model-xyz`（虚构） | **400** | Model not exist. | 名字无效 |

**文案相同、状态码不同**——这就是判据。`classify_model_probe()` 按状态码 + 文案
联合判定，`tests/test_doctor.py` 把这个区别钉住了。

对照之下，同一个 key 探测 `wanx2.1-imageedit`（纹理编辑）返回的是参数校验报错，
说明模型名被接受、模型可用——这也是 `--edit-backend api` 能真跑通的原因。

两条出路：
1. 去 Model Studio 控制台开通「AI试衣」，然后 `--vton-backend api` 直接可用；
2. 换 [FASHN](https://github.com/fashn-AI/fashn-vton-1.5) 托管接口（`FASHN_API_KEY`），
   端点实测可达（HTTP 401，即缺鉴权而非不可达）。

## 三、权重门槛：18.8 GB，且 HF 仓库是 gated

| 来源 | 实测 | 说明 |
|---|---|---|
| 阿里云 OSS `LHM_prior_model.tar` | HEAD 200，**18.8 GB** | 官方脚本里的下载地址，量大但直连快 |
| HF `3DAIGC/LHM-500M-HF` | 模型页 200，**tree 接口 403** | gated 仓库：需 HF token 并接受条款 |
| `hf-mirror.com` 同仓库 | 同上 | 镜像不绕过 gated，仍需 token |

权重不进仓库（`.gitignore` 已排除 `weights/`），`scripts/fetch_weights.py` 会在下载后
写 `MANIFEST.json` 钉住哈希，避免上游静默更新导致结果漂移。

## 四、网络：会话之间会变，别把某次的结论当常态

同一台机器，两次会话的实测结果相反：

| 目标 | 上次会话 | 本次 |
|---|---|---|
| `github.com` | 502 | 200 |
| `huggingface.co` | 502 | 200 |
| `hf-mirror.com` | 200 | 200 |
| Google Storage（MediaPipe `.task`） | 稳定截断在 8 MiB | 未复测 |
| PyPI | 正常 | 正常（126 MB 轮子 2.8 MB/s） |

教训：**「下载不到」要先分清是「网络不通」还是「环境不兼容」**。本次的 LHM 是后者，
而上一次把前者当成了主因。

## 五、试穿：绕开「账号未开通」的另一条路（本地权重，已在 py3.14 上装通）

试穿不必等账号开通——FASHN VTON v1.5 权重是 Apache-2.0，且它的要求比 LHM 宽松
（`requires-python >=3.10`，只依赖 torch / onnxruntime，没有 `chumpy`、`decord`
这类停更包）。实测在 **Python 3.14.5 上装通并通过导入验证**：

```bash
# 1) CPU 版 torch（126.7 MB；CUDA 版 2.4 GB，8 GB 显存 + 磁盘紧张时先用 CPU）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2) 上游包本体：--no-deps 是必须的，它钉的是 onnxruntime-gpu
git clone https://github.com/fashn-AI/fashn-vton-1.5.git
pip install -e ./fashn-vton-1.5 --no-deps
pip install onnxruntime fashn-human-parser

# 3) 权重：ModelScope 单连接 2.5 MB/s，HF 直连 0.15 MB/s
#    model.safetensors 1.94 GB + yolox_l.onnx 217 MB + dw-ll_ucoco_384.onnx 134 MB
python scripts/pdl.py "https://modelscope.cn/api/v1/models/fashn-ai/fashn-vton-1.5/repo?Revision=master&FilePath=model.safetensors" weights/fashn_vton/model.safetensors
python scripts/pdl.py "https://modelscope.cn/api/v1/models/fashn-ai/DWPose/repo?Revision=master&FilePath=yolox_l.onnx" weights/fashn_vton/dwpose/yolox_l.onnx
python scripts/pdl.py "https://modelscope.cn/api/v1/models/fashn-ai/DWPose/repo?Revision=master&FilePath=dw-ll_ucoco_384.onnx" weights/fashn_vton/dwpose/dw-ll_ucoco_384.onnx
```

三个坑，都是实测踩出来的：

1. **HF 直连会被中间代理破坏**——返回缺 `X-Repo-Commit` 头，
   `huggingface_hub` 直接报「不像 Hugging Face 端点」。设
   `HF_ENDPOINT=https://hf-mirror.com`，或直接从 ModelScope 取文件。
2. **并发不是加速而是限流**：ModelScope 上 8 线程实测 0.27 MB/s，
   单连接 1.17–2.49 MB/s。国内 CDN 按连接限速，多开连接只会互相挤。
3. **`pip install` 在本机会报 `PermissionError: WinError 5`**——
   WorkBuddy 的 `sitecustomize` 拦截了 pip 的写权限探测（临时文件删不掉）。
   加 `--user` 可绕过。

### 但 CPU 跑不动：实测每步 5 分 25 秒

整条链路在 CPU 上确实跑起来了（模型加载、DWPose、人体解析、开始采样），
但采样阶段的实测速度是**每步 325 秒**，30 步需要约 **2 小时 40 分**。
这个数字决定了 CPU 只能用来验证链路，不能用来出图。

### GPU 路线：装 cu126 轮子后 49 秒出图（已验证）

cp314 的 CUDA 轮子最高到 `torch 2.14.1+cu126`（`2.15.0` 只有 CPU 版），
配对 `torchvision==0.29.1+cu126`。装完实测（RTX 4060 Laptop 8 GB，bf16）：

| 阶段 | CPU（2.15.0+cpu） | GPU（2.14.1+cu126） |
|---|---|---|
| 采样 30 步 | 每步 325 s，共 ~2.6 h | 每步 **1.64 s**，共 **49 s** |
| 端到端（含模型加载/姿态/解析） | 不可用 | **171 s** |
| 产出 | — | 576×864，换装自然、人脸姿态保留 |

CUDA 轮子 2.62 GB，`pip` 经代理下载会假死（临时文件长时间 0 字节），
用 `scripts/pdl.py` 断点直取即可：

```bash
python scripts/pdl.py "https://mirrors.aliyun.com/pytorch-wheels/cu126/torch-2.14.1%2Bcu126-cp314-cp314-win_amd64.whl" outputs/torch-2.14.1+cu126-cp314-cp314-win_amd64.whl 4
pip install --force-reinstall --no-deps "outputs/torch-2.14.1+cu126-cp314-cp314-win_amd64.whl"
pip install --force-reinstall --no-deps "torchvision==0.29.1+cu126" --index-url https://download.pytorch.org/whl/cu126
```

镜像带宽实测：阿里云 1.25 MB/s（4 线程聚合 ~2 MB/s）、上交 0.93、
官方 0.32。wheel 文件名必须保持规范名，`pip install` 才认。

验证命令与判定标准见 `scripts/smoke_vton.py`（产出与原图差异 < 2/255 判为假成功）。
DWPose 的 onnx 在 CPU 上跑几秒即可，无需为它装 onnxruntime-gpu。

## 六、所以现在能跑什么

| 档位 | 现状 | 缺什么 |
|---|---|---|
| `mock` | 可跑 | 无 |
| `proxy`（本地基线） | 可跑，46 个测试覆盖 | 无 |
| `proxy --edit-backend api` | 可跑（wanx 真掩码端点） | 已具备 |
| `proxy --views-backend api` | 可跑 | 已具备 |
| **试穿（本地权重，GPU）** | **可跑，实测 171 s/张，出图质量良好** | 已具备 |
| 试穿（aitryon / FASHN 托管） | 不可跑 | 账号开通 或 `FASHN_API_KEY` |
| `real`（LHM 重建） | 不可跑 | Python 3.12 环境 + CUDA torch + 18.8 GB 权重 + `gsplat` 编译 |

把这张表和 `doctor` 的输出对齐，就不会再出现「代码明明写完了却跑不起来」的困惑。
