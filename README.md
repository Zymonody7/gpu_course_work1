# 2025年秋季国科大《GPU架构与编程》作业一
# https://github.com/Zymonody7/gpu_course_work1
- **定位**：Fashion-MNIST 上的脉冲卷积网络 (SCNN) 训练与 CUDA 推理示例。
- **主要文件**：
  - `train.py`：使用 SpikingJelly + PyTorch 训练 SCNN，并在达到精度阈值时将权重导出为 txt。
  - `inference.cu`：CUDA/WMMA 推理实现，接受导出的权重文件并在 GPU 上批量计算。
  - `weights.zip`：预导出的模型参数（conv1/conv2/fc1/fc2/fc3 的 weight 与 bias）。

## 环境依赖
- Python 3.x，`torch`, `torchvision`, `spikingjelly`，以及 `numpy`.
- CUDA 工具链（含 `nvcc`），可用的 NVIDIA GPU 与驱动。

## 训练
1. 安装依赖：`pip install torch torchvision spikingjelly numpy`.
2. 运行 `python train.py`（自动下载 Fashion-MNIST 到 `work1/data`）。
3. 最佳模型参数会保存在 `work1/weights_tuned/*.txt`，格式与推理程序一致。

## 推理
1. 准备权重：解压 `weights.zip` 或使用自己训练得到的 `weights_tuned` 目录。
   ```bash
   unzip weights.zip -d weights
   ```
2. 确保数据存在：`train.py` 下载的原始二进制放在 `work1/data/FashionMNIST/raw/`；推理程序会从 `<权重目录>/../../..` 路径查找该原始数据。
3. 编译：
   ```bash
   nvcc -O3 -std=c++17 inference.cu -o inference
   ```
4. 运行（参数为权重目录路径）：
   ```bash
   ./inference weights
   ```
   输出格式：`<推理耗时秒>:<准确率>`，例如 `0.1234:0.9123`。

## 推理实现要点（inference.cu）
- **输入/权重加载**：从权重目录读取 txt 并拷贝到 GPU；从 `data/FashionMNIST/raw` 直接读取原始二进制测试集，并做 `((px/255-0.5)/0.5)` 归一化。
- **算子覆盖**：两层 5x5 conv + 2x2 maxpool + 三层全连接，IF 脉冲激活在 GPU 侧完成；全流程在单个 `scnn_inference` 中完成。
- **Tensor Core WMMA**：线性层和部分卷积使用 `wmma::mma_sync`（`WMMA_M/N/K=16`）进行半精度矩阵乘加，累加精度为 float；共享内存双缓冲（`s_a/s_b`）减少全局访存。
- **线程/块组织**：`BLOCK_ROWS=BLOCK_COLS=64`，`THREADS_PER_BLOCK=512`，warp 粒度切 16x16 瓦片；对非对齐边界使用显式 padding/clamp 保证安全访存。
- **多流并行**：`num_streams=2`，每个流处理一个 mini-batch（默认 batch=128）；CPU 侧为每个流分配 pinned host 缓冲，利用 `cudaMemcpyAsync` 与 kernel 重叠。
- **后处理**：每批次在 GPU 上累加输出，拷回 host 后在 CPU 上取 `argmax` 得到预测；最终与标签比对计算精度，并输出耗时与精度。
- **错误处理**：所有 CUDA API 包装在 `checkCudaErrors` 中，确保失败立即报错退出。

## 目录结构示例
```
work1/
├── inference.cu
├── train.py
├── weights.zip
├── weights/            # 解压后的权重（运行推理用）
└── data/FashionMNIST/  # train.py 自动下载
```
