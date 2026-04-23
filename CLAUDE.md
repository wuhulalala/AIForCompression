# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# AIForCompression 项目

对多种学习型压缩模型在 ERA5 气象数据集上进行统一评测。

## 项目结构

```
AIForCompression/
├── models/                         # 所有模型源码统一放在这里
│   ├── CAESAR/                     # CAESAR 科学数据压缩模型
│   ├── CRA5/                       # CRA5 工具库（mean/std 归一化参数 + CompressAI 预训练模型）
│   ├── DCAE/                       # DCAE 图像压缩模型
│   ├── DCMVC/                      # DCMVC 视频压缩模型（CVPR 2025）
│   ├── DCVC/                       # DCVC / DCVC-RT / DCVC-family 视频压缩模型
│   ├── FLAVC/                      # FLAVC 视频压缩模型
│   ├── LIC-HPCM/                   # LIC-HPCM 图像压缩模型
│   ├── LIC_TCM/                    # LIC-TCM 图像压缩模型
│   ├── RwkvCompress/               # RwkvCompress / LALIC 图像压缩模型
│   └── WeConvene/                  # WeConvene 模型 (ECCV 2024)
├── checkpoints/                    # 所有模型权重统一按 checkpoints/<模型名>/... 存储
│   ├── bmshj2018-factorized/       # CompressAI zoo 权重（mse / ms-ssim）
│   ├── bmshj2018-hyperprior/       # CompressAI zoo 权重（mse / ms-ssim）
│   ├── cheng2020-anchor/           # CompressAI zoo 权重（mse / ms-ssim）
│   ├── cheng2020-attn/             # CompressAI zoo 权重（mse / ms-ssim）
│   ├── dcae/                       # DCAE MSE 权重
│   ├── dcvc-rt/                    # DCVC-RT 2025 权重（cvpr2025_image/video）
│   ├── dcmvc/                      # DCMVC 权重
│   ├── lic-hpcm/                   # LIC-HPCM MSE 权重（hpcm-base / hpcm-large）
│   ├── lictcm/                     # LIC-TCM MSE 权重
│   ├── mbt2018/                    # CompressAI zoo 权重（mse / ms-ssim）
│   ├── mbt2018-mean/               # CompressAI zoo 权重（mse / ms-ssim）
│   ├── rwkvcompress/               # RwkvCompress / LALIC MSE 权重
│   ├── vaeformer-pretrained/       # CRA5 VAEformer 权重
│   ├── weconvene/                  # WeConvene MSE 权重
│   └── <新模型名>/                 # 新模型权重目录
├── Data/
│   └── ERA5/2024/        # 测试数据（pressure.nc + single.nc 配对）
├── results_selected_channels/  # 6通道测试结果
├── unified_results/            # 268通道统一结果和图表
├── test_selected_channels.py   # 统一测试脚本（6目标通道）
├── plot_*.py                   # 各类绘图脚本
└── CLAUDE.md
```

## ERA5 数据说明

- 268 个通道：7 个气压层变量 x 37 个气压层 + 9 个地面变量
- 数据文件格式：`{timestamp}_pressure.nc` + `{timestamp}_single.nc` 配对
- 数据路径：`Data/ERA5/2024/`
- 归一化参数来自 `models/CRA5/cra5/api/mean_std.json` 和 `mean_std_single.json`

## 通道不匹配处理方法

模型通常只支持 3 通道输入。处理方式（参照 DCAE/test_era5.py）：
1. 将 268 通道按 GROUP_SIZE=3 分组（最后一组不足 3 通道则复制最后一个通道补齐）
2. 每组做 minmax 归一化到 [0,1]，送入模型
3. 模型输出后反归一化恢复原始尺度
4. 拼接所有组的结果

## 两种测试模式

1. **268 通道全量测试**：每个模型目录下的 `test_era5.py`，结果在 `models/<模型>/results_era5/`
2. **6 目标通道测试**：`test_selected_channels.py`，只测 z500/t850/v10/u10/t2m/msl（2 组 x 3 通道），结果在 `results_selected_channels/`
   - CRA5 仍用全 268 通道，然后提取 6 通道计算指标
   - 支持 `--models` 过滤：`python test_selected_channels.py --models LIC_TCM`
   - 有 resume 支持，error 条目会自动重跑

## 已知问题：模型导入冲突

DCAE、LIC_TCM、RwkvCompress、WeConvene 等模型目录中可能都有各自的 `models` 包。在同一进程中顺序加载时，先导入的 `models` 会缓存在 `sys.modules` 中，导致后续模型 import 失败。解决方法：用 `--models` 参数单独跑有冲突的模型，或在 loader 中清理相关 `sys.modules` 并显式设置模型源码路径。

## 模型分类（熵编码方式）

- **非自回归**：bmshj2018-factorized, bmshj2018-hyperprior, mbt2018-mean — 并行解码，速度快
- **自回归**：mbt2018, cheng2020-anchor, cheng2020-attn — 顺序解码（CPU），速度慢但压缩率高
- **专用图像/科学数据模型**：DCAE, WeConvene, LIC_TCM, LIC-HPCM, RwkvCompress, CAESAR, CRA5
- **视频模型**：DCMVC, DCVC/DCVC-RT/DCVC-family, FLAVC

## 绘图脚本

- `plot_selected_channels_time_bar.py`：6 通道 encode/decode 时间柱状图（非 CRA5 除以 2）
- `plot_selected_channels_mse_bar.py`：MSE/RMSE 柱状图（仅 mse metric）
- `plot_enc_dec_time_bar.py`：268 通道 encode/decode 时间柱状图
- `plot_forward_time_bar.py`：forward 推理时间柱状图

注意：绘图时非 CRA5 模型的 6 通道时间需除以 2（实际跑了 2 组 x 3 通道）。

## 添加新模型的标准流程

1. 将模型代码放在 `AIForCompression/models/<模型名>/` 目录下
2. 将 checkpoints 放在 `checkpoints/<模型名>/` 目录下（tar 文件需解压）
3. 在模型目录下创建 `test_era5.py`，遵循以下模式：
   - 复用 ERA5 变量定义、读取、通道分组、归一化逻辑（参考 `models/DCAE/test_era5.py`）
   - 适配模型的加载方式（参考模型自带的 eval 脚本和 README）
   - 输出结果到 `<模型目录>/results_era5/`，格式为 summary.json
4. 在 `test_selected_channels.py` 中添加 loader 函数和 config
5. 编写对应的 `run.sh` 提交脚本

## HPC 集群环境

- 提交命令：`sbatch -p gpu_5090 ./run.sh`
- GPU 分区：gpu_5090（RTX 5090）
- Slurm 作业脚本中不要手动 `unset`、`export` 或覆盖 `CUDA_VISIBLE_DEVICES`。该变量由集群调度器根据分配到的 GPU 设置，脚本里改它会破坏 Slurm 的 GPU 绑定。

## run.sh 编写规则

每个模型的 run.sh 必须根据该模型自身的运行方式编写。编写前先阅读模型的 README 和 eval/test 脚本。

固定部分（所有模型通用）：
```bash
#!/bin/bash
#SBATCH --job-name=<模型名>_era5
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=<模型名>_era5_%j.log
#SBATCH --error=<模型名>_era5_%j.log

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh
```

可变部分（根据模型调整）：
- `cd` 到模型对应的目录
- python 命令、脚本名、参数都按模型自身的用法来
- checkpoints 路径统一为 `checkpoints/<模型名>/...`
- 数据路径统一为 `Data/ERA5/2024`
- 如果底层 Python 脚本有 `--gpu` 参数，Slurm 提交脚本默认不要传；让 `CUDA_VISIBLE_DEVICES` 保持调度器分配的值。

## 权重目录约定

- 模型源码只放在 `models/<模型名>/`，不要再把模型源码放在项目根目录。
- 预训练权重统一放在项目根目录的 `checkpoints/` 下，不放进各模型源码目录，除非上游脚本强依赖相对路径且无法通过参数指定。
- README 中标注为 MSE 的权重按模型分目录保存：
  - DCAE：`checkpoints/dcae/mse_*.pth.tar`
  - LIC_TCM：`checkpoints/lictcm/*.pth.tar`
  - LIC-HPCM：`checkpoints/lic-hpcm/hpcm-base/mse/*.pth.tar` 和 `checkpoints/lic-hpcm/hpcm-large/mse/*.pth.tar`
  - RwkvCompress/LALIC：`checkpoints/rwkvcompress/mse/lalic-q*.pth`
  - WeConvene：`checkpoints/weconvene/*.pth.tar`
- DCVC-RT 2025 权重放在 `checkpoints/dcvc-rt/`，主 README 使用 `cvpr2025_image.pth.tar` 和 `cvpr2025_video.pth.tar`。
- DCMVC P-frame 权重放在 `checkpoints/dcmvc/dcmvc_p_frame.pth.tar`；I-frame 权重来自 DCVC-DC 的 `cvpr2023_image_psnr.pth.tar`。
- 视频模型如果上游 README 使用 `./checkpoints` 或 `./ckpt`，优先在运行脚本中传入项目根目录下的统一权重路径；只有在脚本写死相对路径时才创建软链接。

## 代码规范

- Python 3.10，依赖 compressai==1.2.6, xarray, torch
- 除此之外应该递归查看对应模型中的文件，检查哪些依赖没有安装
- 测试脚本统一使用 argparse，支持 `--data_root`, `--ckpt_dir`, `--checkpoint`, `--gpu`, `--compress` 等参数
- 工具脚本统一放在utils目录下，例如AIForCompression放在'/data/run01/scxj523/zsh/project/AIForCompression/utils' 下 里面包含了计算参数量 的脚本，计算mead,std的脚本等等
- 测试脚本一般放在scripts目录下，例如/data/run01/scxj523/zsh/project/AIForCompression/scripts 下

## 测试指标
- 结果必须包含以下指标：
- 参数量
- mse
- rmse
- psnr
- bpp
- compression_ratio
- encode_time_avg
- decode_time_avg
- encode_throughput
- decode_throughput


## 关键路径

- 项目根目录：`/data/run01/scxj523/zsh/project/AIForCompression`
- 数据目录：`/data/run01/scxj523/zsh/project/AIForCompression/Data/ERA5/2024`
- Checkpoints：`/data/run01/scxj523/zsh/project/AIForCompression/checkpoints/`
- 模型源码目录：`/data/run01/scxj523/zsh/project/AIForCompression/models/`
- CRA5 归一化参数：`models/CRA5/cra5/api/mean_std.json`, `models/CRA5/cra5/api/mean_std_single.json`
