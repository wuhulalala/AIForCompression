# 压缩 Pipeline 使用说明

这一层把数据集处理和模型执行拆开：

```text
Dataset Adapter -> CanonicalSample -> Model View -> Codec Runner -> Metrics
```

当前支持的数据集：

- `kodak`：RGB 图像目录，输出 canonical `[C, H, W] uint8`。
- `era5`：配对的 `{timestamp}_pressure.nc` 和 `{timestamp}_single.nc`，输出 canonical `[C, H, W] float32`。

## 准备 Canonical 数据

Kodak：

```bash
python scripts/run_compression_pipeline.py \
  --dataset kodak \
  --data_root /data/run01/scxj523/zsh/project/Data/Kodac \
  --output_dir prepared_datasets/kodak \
  --max_samples 24
```

ERA5：

```bash
python scripts/run_compression_pipeline.py \
  --dataset era5 \
  --data_root /data/run01/scxj523/zsh/project/Data/ERA5/2024 \
  --output_dir prepared_datasets/era5_2024 \
  --max_samples 1
```

输出内容包括：

- `dataset_manifest.json`
- `samples.jsonl`
- 每个 canonical sample 对应一个 `.npy` 数组

## 运行图像类压缩

对于暴露 CompressAI-like API 的模型：

```python
compressed = model.compress(x)
reconstructed = model.decompress(compressed["strings"], compressed["shape"])
```

运行：

```bash
python scripts/run_dataset_compression.py \
  --dataset kodak \
  --data_root /data/run01/scxj523/zsh/project/Data/Kodac \
  --output_dir unified_results/kodak_image_models \
  --models DCAE LIC_TCM \
  --max_samples 2
```

Slurm 提交全权重、全 Kodak sample 的 framework smoke 任务时使用：

```bash
sbatch scripts/run_framework_smoke_model.sh kodak_dcae all all
```

第三个参数 `all` 会传 `--max_samples -1`，遍历 Kodak 目录下全部 sample。集群作业脚本不要手动 `unset`、`export` 或覆盖 `CUDA_VISIBLE_DEVICES`；该变量由 Slurm 按分配到的 GPU 设置，脚本改动会破坏 GPU 绑定。

ERA5 使用同一个 runner。View builder 会按 3 个通道一组分组；最后一组不足 3 个通道时，重复最后一个真实通道进行补齐；每个通道归一化到 `[0, 1]`；计算指标前再执行反变换：

```bash
python scripts/run_dataset_compression.py \
  --dataset era5 \
  --data_root /data/run01/scxj523/zsh/project/Data/ERA5/2024 \
  --output_dir unified_results/era5_image_models \
  --models DCAE LIC_TCM \
  --max_samples 1
```

## 运行 CRA5

CRA5 已通过其原生 268 通道 ERA5 路径接入。它只兼容保留完整 268 个通道的 ERA5 canonical sample：

```bash
python scripts/run_dataset_compression.py \
  --dataset era5 \
  --data_root /data/run01/scxj523/zsh/project/Data/ERA5/2024 \
  --output_dir unified_results/era5_cra5 \
  --models CRA5 \
  --max_samples 1
```

使用 CRA5 时不要传 `--max_channels`。Runner 会以 `x.shape == [1, 268, H, W]` 调用 `model.compress(x)`，然后调用 `model.decompress(strings, z_shape)`。

## 运行 CAESAR

CAESAR 通过自己的原生 model view 接入，不走 image-group runner。它需要 ERA5 的连续时间段 stack：

```text
[V, T, H, W] -> [V, S, T, H, W]
```

其中 `S=1`，`T` 是连续时间帧数：

- `caesar_v`：需要至少 8 个连续 timestamp。
- `caesar_d`：需要至少 16 个连续 timestamp。

示例：

```bash
python scripts/run_dataset_compression.py \
  --dataset era5 \
  --data_root /data/run01/scxj523/zsh/project/Data/ERA5/2024 \
  --output_dir unified_results/era5_caesar_v \
  --models caesar_v \
  --max_samples 8
```

如果要跑 `caesar_d`：

```bash
python scripts/run_dataset_compression.py \
  --dataset era5 \
  --data_root /data/run01/scxj523/zsh/project/Data/ERA5/2024 \
  --output_dir unified_results/era5_caesar_d \
  --models caesar_d \
  --max_samples 16
```

可以用 `--caesar_start_index` 选择连续窗口的起点。Runner 会检查窗口内 timestamp 是否按固定时间间隔递增，避免把不连续时间片送进 CAESAR。
