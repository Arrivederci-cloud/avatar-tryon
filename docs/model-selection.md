# 模型选型与许可清单（面向公开 GitHub 仓库）

> 选型第一约束不是效果，而是许可证。公开仓库意味着你在**分发代码**，别人要能克隆、安装、跑起来。
> 一个「效果最好但许可证不允许再分发」的模型，会让你的 README 变成不可复现的声明。
>
> 本文所有许可结论基于 2026-10 各项目仓库与模型卡，上线前请以最新许可文件为准，重要商用场景请法务复核。

## 一、核心栈（可随仓库分发、可商用）

| 环节 | 模型 / 库 | 许可证 | 商用 | 随仓分发 | 说明 |
|---|---|---|---|---|---|
| 人体重建与驱动 | **LHM / LHM++** | Apache-2.0 | 可 | 可（代码） | 单图秒级出带蒙皮权重的高斯人体，原生 SMPL-X 驱动 |
| 人体骨架与体尺 | **MHR（Momentum Human Rig）** | Apache-2.0 | 可 | 可 | 解耦骨架与体型，是体尺的几何基准 |
| 三视图纹理补全 | **Qwen-Image-Edit-2509** | Apache-2.0 | 可 | 可 | 用于 UV 空间纹理 inpaint，替代不可再分发的多视图扩散模型 |
| 2D 服装试穿 | **FASHN VTON v1.5** | Apache-2.0 | 可 | 可 | 972M 参数、像素空间、无掩码，目前唯一可直接商用的开源试衣 |
| 2D 发型迁移 | **HairFastGAN** | MIT | 可 | 可（代码与权重） | 亚秒级，代码与 HF 权重均为 MIT，是发型环节最干净的选择 |
| 自动骨骼绑定 | **UniRig** | MIT（代码） | 可 | 可 | 1–5 秒出骨架与蒙皮；人形精度接近人工 |
| Web 渲染 | **three.js r186** | MIT | 可 | 可 | 原生高斯泼溅，与 PBR 网格同场景混排 |
| 高斯 LOD 流式 | **Spark 2.x** | MIT | 可 | 可 | 手机端 50 万溅射体预算，.RAD 分块流式 |
| 图像分割 / 抠图 | **BiRefNet** | MIT | 可 | 可 | 输入质检的前置环节 |

## 二、可选增强（不进仓库，运行时接入，需单独告知用户）

| 环节 | 模型 | 许可证 | 处理方式 |
|---|---|---|---|
| 体尺精度增强 | SAM 3D Body | **SAM License**（Meta，非 OSI 认证） | 允许商用但有 acceptable use 限制、禁止逆向、诉讼即终止授权；**不要 vendored**，让用户自己从官方渠道下载并接受条款 |
| 通用物件 3D | Tripo / Meshy API | 商业 | 通过 API 调用，不涉及代码分发 |
| 布料仿真 | Blender 布料（GPL） | GPL-2.0+ | 只能以**独立进程**调用（`blender --background --python`），不要 `import bpy` 到主程序 |

## 三、明确不要放进仓库

| 模型 | 许可证 | 原因 |
|---|---|---|
| CatVTON / IDM-VTON / OOTDiffusion | CC BY-NC-SA 4.0 | 禁止商用 + 相同方式共享，会把你的仓库传染成非商用 |
| MagicMan / HumanOrbit / CHROME | 未见明确许可 | **没有许可证 = 默认保留所有权利**，法律上不能再分发 |
| HairPort / Stable-Hair | 研究用途 | 同上，仅可作论文对比引用 |
| TRELLIS.2 | 研究许可 | 不可商用再分发 |
| Hunyuan3D-2.1 | 社区许可 | 含月活上限与地域排除条款 |
| CLO3D / Style3D | 商业软件 | 桌面端，无法作为仓库依赖；改为离线预处理并分发产物 |

## 四、四条仓库纪律

1. **许可证用 Apache-2.0，不用 MIT。**
   Apache-2.0 含显式专利授权，公司评估时更容易通过；对 ML 项目比 MIT 更稳妥。

2. **权重一律不入 git。**
   `scripts/fetch_weights.py` 从 HuggingFace / ModelScope 下载，锁定 revision 与 SHA256，
   避开 GitHub 100MB 单文件限制与 LFS 配额，也避免把第三方权重当成自己的资产分发。

3. **模型走适配器接口，实现可替换。**
   `models/base.py` 定义 `reconstruct()` / `try_on()` / `transfer_hair()` 等抽象，
   每个模型一个实现文件。非自由许可的实现不提供代码，只在文档写「自行接入」，
   这样仓库永远保持可分发，增强能力由使用者自行拼装。

4. **示例素材必须有授权。**
   不要把人脸照片提交进仓库——既是隐私合规问题，也会让仓库被投诉。
   示例统一用合成人物、CC0 素材，或自己出镜并明确授权。

## 五、THIRD_PARTY_NOTICES 模板

```
本项目采用 Apache License 2.0。

第三方组件与模型：

1. LHM / LHM++ — Apache License 2.0
   https://github.com/aigc3d/LHM
   权重通过 scripts/fetch_weights.py 从官方渠道获取，本仓库不再分发。

2. MHR (Momentum Human Rig) — Apache License 2.0
   https://github.com/facebookresearch/MHR

3. Qwen-Image-Edit-2509 — Apache License 2.0
   https://huggingface.co/Qwen

4. FASHN VTON v1.5 — Apache License 2.0
   https://github.com/fashn-AI/fashn-vton-1.5

5. HairFastGAN — MIT License
   Copyright (c) AIRI
   https://github.com/AIRI-Institute/HairFastGAN

6. UniRig — MIT License
   https://github.com/VAST-AI-Research/UniRig

7. three.js — MIT License，Spark — MIT License

可选依赖（本项目不分发，由使用者自行安装并遵守其条款）：
- SAM 3D Body — SAM License (Meta Platforms)
- Blender — GPL-2.0+（仅以独立进程调用）
```

## 六、建议目录骨架

```
.
├── LICENSE                  # Apache-2.0
├── THIRD_PARTY_NOTICES.md   # 上方模板
├── README.md                # 含「模型不随仓库分发」的安装说明
├── docs/
│   ├── model-selection.md   # 本文件
│   └── pipeline.md          # 单图 → 三视图 → 试穿 的链路说明
├── models/
│   ├── base.py              # 抽象接口
│   ├── lhm_adapter.py
│   ├── mhr_adapter.py
│   ├── vton_adapter.py      # FASHN VTON
│   └── hair_adapter.py      # HairFastGAN
├── pipeline/
│   ├── quality_check.py     # 分割、关键点、尺度标定
│   ├── three_view.py        # 几何投影 + UV 纹理补全
│   └── garment.py           # 形变目标插值与装配
├── web/                     # three.js + Spark 前端
├── scripts/
│   └── fetch_weights.py     # revision + SHA256 锁定
└── assets/
    └── samples/             # 合成或已授权素材
```

## 七、README 免责声明建议

```
本项目为技术研究与原型验证用途。
- 不提供医疗、诊断或法律意义上的身体测量结论；
- 三视图中侧视图与背视图为模型生成结果，不代表本人真实外观；
- 使用者上传的影像涉及人脸与体型信息，属敏感个人信息，
  部署为对外服务前请自行完成隐私合规评估与算法备案。
```
