# Hyperparameter Optimization for EfficientNet-B3 Training

## 📚 Project File Overview
This module implements **automated hyperparameter optimization** for EfficientNet-B3 fine-tuning using **PyTorch** and **Ray Tune**.  
The goal is to systematically search the space of critical training hyperparameters to maximize model performance while minimizing manual trial-and-error.

---

## 🚀 Key Techniques Implemented
- **Ray Tune Hyperparameter Search** over learning rates, optimizers, batch sizes
- **Asynchronous Hyperband Scheduler** for early stopping of underperforming trials
- **EfficientNet-B3 model training** with DDP (optional) and AMP support
- **Automatic checkpointing and reporting** of best configurations
- **Multi-GPU and CPU-aware parallelization** of trials

---

## 🛠️ Code Structure

| File                       | Description |
|-----------------------------|-------------|
| `hyperparameter-optimization.py` | Main script for tuning hyperparameters and benchmarking results |
| `search_spaces/`            | Defines parameter ranges (e.g., learning rate, optimizer, batch size) |
| `results/`                  | Stores trial metrics, best config logs, and plots |

---

## ⚙️ How to Run

Install Ray Tune first if not already installed:
```bash
pip install ray[tune]
