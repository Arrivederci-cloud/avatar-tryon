# 第三方组件与模型声明

本项目代码采用 Apache License 2.0。以下第三方组件与模型各自保留原许可，
其权利归原作者所有。

本仓库**不分发任何模型权重**。`scripts/fetch_weights.py` 仅从官方渠道下载，
权重存放在 `weights/`（已在 `.gitignore` 中排除）。

## 一、默认依赖（可商用、可再分发）

### MHR (Momentum Human Rig)
- 许可证：Apache License 2.0
- 来源：https://github.com/facebookresearch/MHR
- 用途：参数化人体骨架与体尺

### LHM / LHM++
- 许可证：Apache License 2.0
- 来源：https://github.com/aigc3d/LHM
- 引用：Qiu et al. *LHM: Large Animatable Human Reconstruction Model from a Single Image in Seconds*, ICCV 2025

### Qwen-Image-Edit
- 许可证：Apache License 2.0
- 来源：https://huggingface.co/Qwen
- 用途：UV 空间的纹理补全

### FASHN VTON v1.5
- 许可证：Apache License 2.0
- 来源：https://github.com/fashn-AI/fashn-vton-1.5
- 用途：2D 服装试穿

### HairFastGAN
- 许可证：MIT License
- Copyright (c) AIRI (Artificial Intelligence Research Institute)
- 来源：https://github.com/AIRI-Institute/HairFastGAN
- 引用：Nikolaev et al. *HairFastGAN: Realistic and Robust Hair Transfer with a Fast Encoder-Based Approach*, NeurIPS 2024

### UniRig
- 许可证：MIT License
- 来源：https://github.com/VAST-AI-Research/UniRig
- 引用：*One Model to Rig Them All: Diverse Skeleton Rigging with UniRig*, SIGGRAPH 2025

### three.js
- 许可证：MIT License
- 来源：https://github.com/mrdoob/three.js

### Spark
- 许可证：MIT License
- 来源：https://github.com/sparkjsdev/spark

## 二、可选依赖（本项目不提供、不分发）

以下模型因许可限制或授权不明确，**未包含在本仓库中**。使用者如需启用，
应自行前往官方页面阅读并接受其条款。

### SAM 3D Body
- 许可证：**SAM License**（Meta Platforms，非 OSI 认证）
- 来源：https://github.com/facebookresearch/sam-3d-body
- 注意：允许商用但含 acceptable use 限制；禁止逆向工程；对 Meta 提起诉讼将终止授权。
  本仓库不分发其代码与权重。

### MagicMan / HumanOrbit / CHROME
- 许可证：未见明确开源许可
- 说明：**没有许可证即默认保留所有权利**，法律上不可再分发。仅可作论文对比引用。

### CatVTON / IDM-VTON / OOTDiffusion
- 许可证：CC BY-NC-SA 4.0
- 说明：禁止商业使用，且要求相同方式共享——引入会传染本仓库的整体许可。

### TRELLIS.2
- 许可证：研究用途
- 说明：禁止商用再分发。

### Hunyuan3D
- 许可证：社区许可
- 说明：含月活跃用户上限与地域排除条款。

### CLO3D / Style3D / Browzwear
- 许可证：商业软件
- 说明：不在本仓库中作为依赖。版型仿真在离线环境中完成，产物（形变目标）以数据形式使用。

## 三、数据集与素材

本仓库 `assets/samples/` 下不包含任何真人照片。
示例素材应为合成人物、CC0 素材或已获得明确授权的影像。

## 四、合规提示

本项目处理人脸与体型信息，在中国大陆属**敏感个人信息**。部署为对外服务前，
应完成单独同意、显著标识（深度合成内容）、安全评估与算法备案。
详见 `docs/model-selection.md` 与项目文档中的合规章节。
