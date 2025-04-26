# Distributed Data Parallel (DDP) Training for EfficientNet-B3

## 📚 Project File Overview
This project file implements **Distributed Data Parallel (DDP)** training for the **EfficientNet-B3** model on a **4×A100 GPU** setup.  
The goal is to accelerate deep learning training using **multi-GPU parallelism** and **mixed-precision (AMP)** without sacrificing model accuracy or stability.

---

## 🚀 Key Techniques Implemented
- **PyTorch Distributed Data Parallel (DDP)** using `torchrun`
- **Automatic Mixed Precision (AMP)** for memory and speed optimization
- **EfficientNet-B3 Fine-tuning** on a large multi-class medical image dataset
- **Dynamic batch splitting** across GPUs with synchronized updates
- **Early stopping** and **adaptive learning rate** scheduling (ReduceLROnPlateau)
- **Performance logging** and **model checkpointing**

---

## 🛠️ Code Structure

| File           | Description |
|----------------|-------------|
| `DDP.py`        | Main script to launch distributed multi-GPU training using PyTorch’s DDP module |
| `config/`       | Contains training hyperparameters and runtime settings (optional) |
| `checkpoints/`  | Directory where best model checkpoints are saved |
| `plots/`        | Directory where training and validation performance plots are saved |

---

## ⚙️ How to Run

```bash
torchrun --nproc_per_node=4 DDP.py
