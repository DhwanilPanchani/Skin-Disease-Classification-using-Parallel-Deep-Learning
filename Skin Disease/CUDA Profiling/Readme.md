# CUDA Profiling and Kernel Analysis for Deep Learning Models

## 📚 Project File Overview
This porject file focuses on **CUDA-level profiling** and **GPU kernel analysis** for deep learning training workflows.  
We benchmarked and visualized GPU operations during the training of **EfficientNet-B3** and **ResNet50** models to:
- Identify performance bottlenecks
- Understand kernel-level operation costs
- Optimize resource utilization and training throughput

---

## 🚀 Key Techniques Implemented
- **PyTorch Native Profiler** for capturing CUDA operations
- **Operation-level timing breakdown** (Conv2D, MatMul, ReLU, BatchNorm, etc.)
- **Kernel launch count and duration analysis**
- **Memory usage profiling** during forward and backward passes
- **Dynamic bar plots and time distribution charts** for easy visualization

---

## 🛠️ Code Structure

| File                     | Description |
|---------------------------|-------------|
| `cuda-profiling.py`        | Runs model training with profiler hooks enabled; records detailed CUDA operation data |
| `cuda-profiling-plots.py`  | Generates visualizations (bar charts) from the profiler output |
| `plots/`                  | Folder containing generated profiling plots for analysis |

---

## ⚙️ How to Run

1. **Profile Model Training**
   ```bash
   python cuda-profiling.py
