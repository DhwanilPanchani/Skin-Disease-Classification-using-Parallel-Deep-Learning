# Parallel Deep Learning for Skin Disease Classification

## 📚 Project Overview
This project explores high-performance deep learning techniques for large-scale **skin disease classification** using **EfficientNet-B3**, trained on a **35-class medical image dataset**.  
We leverage **multi-GPU Distributed Data Parallel (DDP)** training, **mixed-precision (AMP)**, and **data pipeline optimizations** to accelerate training and achieve high accuracy without sacrificing stability or generalization.

---

## 🚀 Key Techniques Implemented
- **EfficientNet-B3** fine-tuning with ImageNet weights
- **DistributedDataParallel (DDP)** for multi-GPU scaling across 4×A100 GPUs
- **Automatic Mixed Precision (AMP)** training for faster throughput and reduced memory usage
- **Parallel data loading strategies** to optimize CPU/GPU utilization
- **CUDA profiling** to analyze kernel bottlenecks and operation breakdown
- **Dynamic learning rate scheduling** using `ReduceLROnPlateau`
- **Early stopping** and **checkpointing** based on validation performance

---

## 🛠️ Code Structure

| File/Folder                | Description |
|-----------------------------|-------------|
| `Model Training.ipynb`       | Full training pipeline for EfficientNet-B3 (DDP + AMP) |
| `DDP.py`                     | Script for launching distributed training with `torchrun` |
| `dataparallel.py`            | Benchmark different data loading and multi-GPU processing strategies |
| `cuda-profiling.py`          | CUDA kernel profiling script (operations and memory analysis) |
| `cuda-profiling-plots.py`    | Plotting scripts for CUDA profiling visualizations |
| `plots/`                     | Folder containing all generated analysis plots |
| `DEMO.zip`, `DEMO-2.zip`     | Complete code archive for reproducibility |

---

## 📊 Dataset

- **Dataset Name:** Massive Skin Disease Balanced Dataset (Kaggle)
- **Size:** ~245,000 images across 35 classes
- **Data Split:**
  - 70% Training
  - 30% Testing
  - 10% of training data used for validation
- **Preprocessing:**
  - Resizing to 256×256
  - Random crops to 224×224
  - Random horizontal flipping
  - Normalization using ImageNet mean and std

---

## ⚙️ Training Details

| Aspect              | Configuration |
|----------------------|---------------|
| Model                | EfficientNet-B3 (pretrained) |
| GPUs                 | 4× A100 40GB GPUs |
| Parallelism          | DistributedDataParallel (DDP) |
| Mixed Precision      | Enabled (AMP with GradScaler) |
| Optimizer            | Adam (lr=0.001) |
| Loss Function        | CrossEntropyLoss |
| Batch Size           | 64 per GPU (Global batch size = 256) |
| Learning Rate Scheduler | ReduceLROnPlateau |
| Early Stopping       | Patience = 3 epochs (on validation loss) |

---

## 🧪 Results

### ✅ Final Performance:
- **Training Accuracy:** ~98%
- **Validation Accuracy:** ~96.8%
- **Test Accuracy:** ~96.79%
- **Micro Average AUC:** ~0.999
- **Training Time Reduction:** ~3.3× faster with 4 GPUs (vs 1 GPU)

### ✅ Throughput Gains:
- **Single GPU (FP32 vs AMP):** ~17% speedup with AMP
- **4 GPUs (FP32 vs AMP):** AMP slightly outperforms FP32 throughput (~178 vs ~160 img/s)

### ✅ Memory Efficiency:
- **Training Memory (Batch 64):**
  - FP32: ~8200 MB
  - AMP: ~4100 MB
- **Inference Memory:**
  - FP32: ~300 MB
  - AMP: ~150 MB

### ✅ Parallel Data Loading:
- Multi-GPU DataLoader and Manual splitting achieved ~3.6× speedup compared to single-process simple loading.

---

## 📈 Key Visualizations

- Training vs Validation Accuracy and Loss Curves
- CUDA Operation Time Breakdown (EfficientNet-B3)
- CUDA Kernel Launch Count and Average Duration
- Speedup vs Number of GPUs
- Scaling Efficiency across 1, 2, and 4 GPUs
- Parallel Data Loader Timing Comparison

> 📂 All plots are available in the `plots/` directory.

---

## 📋 CUDA Profiling Insights

- **Top Operations:** `conv2d`, `matmul`, `relu`
- **Most Time-Consuming Kernels:** convolution and batch normalization
- **Bottlenecks Identified:** 
  - Gradient synchronization (all-reduce) at 4 GPUs
  - Data loading delays minimized by multi-threaded CPU pipelines
- **Optimization Success:** No divergence with AMP/DDP; smooth convergence.

---

## 🔥 Key Takeaways
- **Multi-GPU parallelism** reduced training time from ~59 minutes to ~17.5 minutes per epoch.
- **Mixed Precision** (AMP) improved memory utilization and training throughput without loss in accuracy.
- **Efficient data pipeline** ensured >90% GPU utilization with no bottleneck.
- **Scalable design** suitable for larger models (EfficientNet-B5, B7) and larger datasets in the future.

---

## 🔮 Future Work
- Scaling beyond 4 GPUs across multiple nodes
- Training larger models (EfficientNet-B5, B7)
- Deploying inference optimized models via TensorRT or ONNX
- Applying similar techniques to multi-modal medical datasets

---


