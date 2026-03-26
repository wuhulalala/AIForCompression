# Fork CompressAI 扩展为 AI 压缩统一 Benchmark 框架

## 背景

当前 `AIForCompression` 项目在 ERA5 气象数据上评测多种学习型压缩模型，但存在以下问题：
- 各模型代码分散在独立目录，无统一接口
- 模型间存在 `sys.modules` 导入冲突
- benchmark 逻辑重复（每个模型各写一套 test_era5.py）
- 数据加载、归一化、通道分组等逻辑散落各处

## 目标

Fork CompressAI，在其基础上：
1. 将现有模型（DCAE、WeConvene、LIC_TCM、CRA5）适配为 CompressAI 的 `CompressionModel` 子类
2. 设计通用数据集接口，第一版实现 ERA5
3. 构建统一 benchmark 框架（借鉴 nvCOMP 的 benchmark harness 模式）
4. 消除导入冲突和代码重复

## 项目结构

```
compressai-bench/                     # fork 自 CompressAI
├── compressai/
│   ├── models/
│   │   ├── ...                       # CompressAI 原有模型 (bmshj2018, mbt2018, cheng2020...)
│   │   ├── dcae.py                   # DCAE 适配为 CompressionModel 子类
│   │   ├── weconvene.py              # WeConvene 适配
│   │   └── lic_tcm.py                # LIC-TCM 适配
│   ├── datasets/
│   │   ├── base.py                   # 通用数据集基类
│   │   ├── era5.py                   # ERA5 实现（268通道分组、归一化、目标通道提取）
│   │   └── image.py                  # CompressAI 原有图像数据集
│   └── zoo/                          # 预训练模型注册表，新增模型的 checkpoint 入口
├── benchmarks/
│   ├── benchmark_common.py           # 统一工具：数据加载、warmup/timing、指标计算、结果输出
│   ├── benchmark_all.py              # 全模型 benchmark 入口
│   ├── benchmark_single.py           # 单模型 benchmark
│   └── plot/                         # 绘图脚本
├── checkpoints/                      # 预训练权重（按模型名组织）
└── data/                             # 数据集目录
```

## 模型适配设计

### 统一接口

所有模型继承 CompressAI 的 `CompressionModel`，必须实现：

```python
class MyModel(CompressionModel):
    def forward(self, x: Tensor) -> dict:
        """训练用前向传播，返回 {"x_hat": ..., "likelihoods": ...}"""

    def compress(self, x: Tensor) -> dict:
        """编码，返回 {"strings": [...], "shape": ...}"""

    def decompress(self, strings, shape) -> dict:
        """解码，返回 {"x_hat": ...}"""
```

### 各模型适配要点

- **CompressAI 内置模型**（bmshj2018-factorized, bmshj2018-hyperprior, mbt2018, mbt2018-mean, cheng2020-anchor, cheng2020-attn）：无需改动，已实现上述接口。通过 `compressai.zoo` 加载预训练权重。

- **DCAE**：将 `DCAE/models/` 下的网络定义包装为 `CompressionModel` 子类。DCAE 使用自定义的编解码流程，需在 `compress()` / `decompress()` 中适配。

- **WeConvene**：将 `WeConvene/model/` 下的网络包装。WeConvene 有 3D 卷积变体，需注意输入维度处理。

- **LIC_TCM**：将 `LIC_TCM/models/` 下的 TCM 模型包装。已有 CompressAI 风格的接口，适配工作量较小。

- **CRA5**：CRA5 内部已 vendor 了 CompressAI，其模型（vaeformer 等）已基于 CompressAI 构建。需要将其从 vendor 目录提取到 fork 的模型注册表中。CRA5 使用全 268 通道，不走分组逻辑。

## 数据集设计

### 通用基类

```python
class BaseDataset(ABC):
    @abstractmethod
    def __getitem__(self, idx) -> dict:
        """返回 {"data": Tensor, "metadata": dict}"""

    @abstractmethod
    def get_normalization(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 (mean, std) 用于归一化/反归一化"""
```

### ERA5 实现

```python
class ERA5Dataset(BaseDataset):
    """
    - 读取 pressure.nc + single.nc 配对
    - 268 通道：7 气压层变量 x 37 层 + 9 地面变量
    - 归一化参数来自 CRA5 的 mean_std.json
    """
```

### 通道分组 Wrapper

```python
class ChannelGroupWrapper:
    """
    将多通道数据按 GROUP_SIZE=3 分组，送入只支持 3 通道的模型。
    最后一组不足 3 通道则复制最后一个通道补齐。
    输出时裁剪回原始通道数。

    CRA5 等原生支持多通道的模型不使用此 wrapper。
    """
```

这个 wrapper 在 benchmark 层使用，模型本身不感知分组逻辑。

## Benchmark 框架设计（借鉴 nvCOMP）

### 核心流程

类似 nvCOMP 的 `benchmark_template_chunked.cuh`：

```
1. 解析参数（模型名、数据路径、质量等级、GPU 设备等）
2. 加载数据集 + 归一化
3. 加载模型 + checkpoint
4. Warmup（1 轮，不计时）
5. 多轮 compress/decompress 计时（默认 5 轮取平均）
6. 计算指标：PSNR、MSE/RMSE、BPP、压缩比、encode/decode 时间、参数量
7. 输出结果（JSON + 可选 CSV）
```

### benchmark_common.py

```python
# 统一工具函数
def warmup(model, sample_input): ...
def timed_compress(model, x, num_rounds=5): ...
def timed_decompress(model, compressed, num_rounds=5): ...
def compute_metrics(original, reconstructed, compressed_size): ...
def save_results(results, output_path, format='json'): ...
def pad_to_multiple(x, multiple=128, mode='reflect'): ...
def crop_to_original(x, original_shape): ...
```

### 运行方式

```sh
# 跑单个模型
python -m benchmarks.benchmark_single \
    --model cheng2020-attn --quality 6 \
    --dataset era5 --data_root data/ERA5/2024 \
    --gpu 0

# 跑所有模型
python -m benchmarks.benchmark_all \
    --dataset era5 --data_root data/ERA5/2024 \
    --output_dir results/

# 6 目标通道模式
python -m benchmarks.benchmark_single \
    --model dcae --quality 3 \
    --dataset era5 --data_root data/ERA5/2024 \
    --selected_channels z500,t850,v10,u10,t2m,msl
```

## 指标体系

沿用现有指标，统一输出格式：

| 字段 | 说明 |
|------|------|
| model | 模型名 |
| quality | 质量等级/lambda |
| psnr_db | PSNR (dB) |
| mse | 均方误差 |
| rmse | 均方根误差 |
| bpp | bits per pixel |
| compression_ratio | 压缩比 |
| encode_time_s | 编码时间（秒，多轮平均） |
| decode_time_s | 解码时间（秒，多轮平均） |
| num_params | 模型参数量 |

## 模型分类

保留现有分类，用于 benchmark 结果分析：

- **非自回归**：bmshj2018-factorized, bmshj2018-hyperprior, mbt2018-mean — 并行解码
- **自回归**：mbt2018, cheng2020-anchor, cheng2020-attn — 顺序解码
- **专用模型**：DCAE, WeConvene, LIC_TCM, CRA5

## 实施优先级

1. Fork CompressAI，建立项目骨架
2. 实现 ERA5 数据集 + 通道分组 wrapper
3. 适配 DCAE、LIC_TCM、WeConvene 为 CompressionModel 子类
4. 注册 CRA5 模型
5. 实现 benchmark harness（benchmark_common + benchmark_single）
6. 实现 benchmark_all + 绘图脚本
7. 迁移现有 checkpoints 和验证结果一致性

## HPC 环境

- 提交命令：`sbatch -p gpu_5090 ./run.sh`
- GPU 分区：gpu_5090（RTX 5090）
- Conda 环境：`/data/run01/scxj523/zsh/envs/zsh`
- Python 3.10，核心依赖：compressai, torch, xarray, netcdf4
