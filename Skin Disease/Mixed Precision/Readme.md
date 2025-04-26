# Mixed Precision and Multi-GPU Benchmarking for EfficientNet-B3

This benchmarking script evaluates the training performance of the **EfficientNet-B3** model under different setups:
- Single GPU vs Multiple GPUs
- Standard Precision (FP32) vs Mixed Precision (FP16+FP32)

The experiment is run on a system with **4× NVIDIA A100-SXM4-40GB GPUs** using **PyTorch 2.5.1+cu121**.

---

## 📋 System Information
- **Hostname**: `d3149`
- **Platform**: `Linux-5.14.0-362.13.1.el9_3.x86_64`
- **PyTorch Version**: `2.5.1+cu121`
- **GPU Count**: `4`
- **GPUs**:
  - GPU 0: `NVIDIA A100-SXM4-40GB`
  - GPU 1: `NVIDIA A100-SXM4-40GB`
  - GPU 2: `NVIDIA A100-SXM4-40GB`
  - GPU 3: `NVIDIA A100-SXM4-40GB`

---

## 🧠 Model Details
- **Model**: EfficientNet-B3
- **Total Parameters**: `10,750,027`
- **Model Size**: `41.01 MB`

---

## 🚀 Benchmark Results

### Single GPU (GPU 0)
**Standard Precision (FP32)**:
- Average Time: **63.06 ms**
- Throughput: **507.48 images/second**
- Effective Batch Size: **32**
- Memory Used: **5467.19 MB**

**Mixed Precision (AMP)**:
- Average Time: **59.03 ms**
- Throughput: **542.13 images/second**
- Effective Batch Size: **32**
- Memory Used: **2809.96 MB**

---

### 2 GPUs (GPU 0 and GPU 1) - Using DataParallel
**Standard Precision (FP32)**:
- Average Time: **137.44 ms**
- Throughput: **465.64 images/second**
- Effective Batch Size: **64**
- Memory Used: **2817.76 MB**

**Mixed Precision (AMP)**:
- Average Time: **161.41 ms**
- Throughput: **396.52 images/second**
- Effective Batch Size: **64**
- Memory Used: **1524.87 MB**

---

### 4 GPUs (GPU 0, 1, 2, and 3) - Using DataParallel
**Standard Precision (FP32)**:
- Average Time: **214.03 ms**
- Throughput: **598.06 images/second**
- Effective Batch Size: **128**
- Memory Used: **1515.05 MB**

**Mixed Precision (AMP)**:
- Average Time: **229.59 ms**
- Throughput: **557.50 images/second**
- Effective Batch Size: **128**
- Memory Used: **856.63 MB**

---

## 📊 Observations

- **Mixed Precision Training** reduces memory usage significantly (almost by half) compared to standard precision.
- **Single GPU performance** is the highest in terms of throughput per device, but **multi-GPU setups** allow scaling with increased batch sizes.
- **4 GPU DataParallel** setup provided the highest total throughput despite some overhead in communication.
- **Memory Efficiency**: Mixed Precision not only speeds up training slightly but also allows larger batch sizes to fit into memory.

---

## 🛠️ Outputs
- **Scaling comparison plots** are saved in: `scaling-1/plots/`

---

## 📷 GPU Setup Visualization
![GPU Setup Diagram](./Mixed-Precision%20GPU%20setup%20Image.png)

---

## 📌 Notes
- This benchmarking uses **PyTorch's automatic mixed precision (AMP)**.
- **DataParallel** is used for simple multi-GPU scaling across the available devices.
- Model and optimizer states were adapted to support FP16 computation safely.

---

