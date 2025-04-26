# Multi-GPU Data Parallelism Script for Skin Disease Classification

This script benchmarks the performance of an **EfficientNet-B3** model using **DataParallel** across multiple GPUs.  
The goal is to analyze how model training scales when moving from single-GPU to multi-GPU setups.

---

## 📋 System and Model Information

- **Framework**: PyTorch
- **Model**: EfficientNet-B3 (Pretrained on ImageNet)
- **Environment**: Linux system with 4 × NVIDIA A100 GPUs
- **Precision Modes Tested**:
  - Standard Precision (FP32)
  - Mixed Precision (AMP)

---

## 🛠️ Benchmarking Setup

- **GPUs Used**:
  - Single GPU (GPU 0)
  - 2 GPUs (GPU 0 and 1)
  - 4 GPUs (GPU 0, 1, 2, and 3)
- **Data Parallelization**: PyTorch's built-in `torch.nn.DataParallel`
- **Batch Sizes**:
  - 32 (Single GPU)
  - 64 (2 GPUs)
  - 128 (4 GPUs)
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam

---

## 🧪 Benchmarking Steps

1. **Model Initialization**:
   - Load EfficientNet-B3 with pretrained weights.
   - Modify final classification layer to match the number of disease classes.

2. **DataLoader Setup**:
   - Loads synthetic or sampled data for fast benchmarking.
   - Batch sizes scaled appropriately with number of GPUs.

3. **Benchmark Process**:
   - Measure training time for a fixed number of iterations.
   - Run separately for:
     - Standard precision (FP32)
     - Mixed precision (AMP)
   - Record:
     - Average iteration time
     - Throughput (images processed per second)
     - Effective batch size
     - Memory consumption per GPU

4. **Logging and Saving**:
   - Results saved into a scaling performance plot directory.
   - Time and throughput printed after each configuration.

---

## 📈 Key Metrics Monitored

- **Average Training Time per Batch**
- **Throughput (images/second)**
- **Effective Batch Size**
- **Memory Usage per GPU**

---

## 📌 Notes

- **DataParallel** replicates the model across available GPUs automatically.
- Mixed precision training improves speed and reduces memory usage.
- Communication overhead between GPUs can affect scaling efficiency.
- Larger batch sizes are crucial to fully utilize multi-GPU setups.

---

## 🚀 Conclusion

This script provides a clear comparison between single GPU and multi-GPU training efficiency for **EfficientNet-B3**.  
It highlights the tradeoffs in throughput, memory usage, and training time when scaling across multiple GPUs, enabling better system resource utilization in future experiments.

---
