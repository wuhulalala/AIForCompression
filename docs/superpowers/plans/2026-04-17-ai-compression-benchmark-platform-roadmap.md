# AI 压缩 Benchmark 平台简短计划

> 这是一份用于汇报和阶段沟通的短计划，不替代正式设计文档。

**目标：** 基于 `CompressAI` 搭建一个面向外部用户的 AI 压缩 benchmark 平台，支持平台内置数据集与用户上传数据，并在单机单卡环境下完成统一评测、结果回传和可视化展示。

**总体思路：** 采用前后端分离、独立调度器和 worker 的结构，但整体部署在一台机器上。用户只能使用平台维护的模型和 checkpoint，保证 benchmark 结果可控、可比、可复现。

**核心技术：** `CompressAI`、PyTorch、Web 前端、Web API、任务调度、关系型数据库、本地工件存储。

---

## 阶段一：明确范围并冻结首版能力

- 明确 V1 不做脚本执行、不做用户 checkpoint 上传、不做 `MS-SSIM`
- 冻结首批模型范围：
  - CompressAI 官方模型
  - `DCAE`、`WeConvene`、`LIC_TCM`、`CRA5`
- 冻结首批数据集范围：
  - 小规模图像验证集
  - 一套标准 ERA5 benchmark 数据

## 阶段二：搭建平台底座

- 建立用户、模型、数据集、任务、工件等元数据表
- 实现任务状态流转：
  - `validating`
  - `queued`
  - `preparing`
  - `running`
  - `succeeded`
  - `failed`
- 建立上传、结果、日志的统一存储结构
- 打通单 worker、单 GPU 的任务执行链路

## 阶段三：完成 benchmark 引擎

- 实现图像数据集 adapter
- 实现 ERA5 数据集 adapter
- 实现 CompressAI 官方模型 adapter
- 实现科研模型 adapter
- 统一 benchmark 执行流程：
  - warmup
  - 压缩计时
  - 解压计时
  - 指标计算
  - 工件导出



## 阶段四：完善结果质量并准备上线

- 输出统一结果文件：
  - `summary.json`
  - `metrics.csv`
  - `run.log`
- 增加图像重建对比和 ERA5 误差可视化
- 固定 V1 核心指标：
  - `mse`
  - `rmse`
  - `psnr`
  - `bpp`
  - `compression_ratio`
  - `encode_time_avg`
  - `decode_time_avg`
  - `encode_throughput`
  - `decode_throughput`

## 预期产出

外部用户可以：

- 选择平台内置数据集或上传自己的数据
- 在受控模型集合中选择压缩器
- 通过网页提交 benchmark 任务
- 获得统一指标、吞吐、日志和结果下载

