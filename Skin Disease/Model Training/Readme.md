# Model Training for Skin Disease Classification

This notebook implements the training pipeline for fine-tuning an **EfficientNet-B3** model on a custom **skin disease dataset**.  
The training leverages data augmentation, weighted loss for class imbalance, and mixed precision (AMP) to improve performance.

---

## 📋 System and Model Information

- **Framework**: PyTorch
- **Model**: EfficientNet-B3 (Pretrained backbone)
- **Dataset**: Custom Skin Disease Images
- **Training Techniques**:
  - Transfer Learning
  - Data Augmentation
  - Mixed Precision Training (AMP)
  - Weighted Cross-Entropy Loss

---

## 🛠️ Training Setup

- **Optimizer**: Adam
- **Learning Rate**: 1e-4
- **Loss Function**: CrossEntropyLoss with class weights
- **Batch Size**:
  - 32 (single GPU)
  - 64 (2 GPUs with DataParallel)
  - 128 (4 GPUs with DataParallel)
- **Epochs**: 20 (can be adjusted based on convergence)
- **Mixed Precision**: Enabled via `torch.cuda.amp`

---

## 🧪 Training Pipeline

1. **Dataset Preparation**:
   - Custom `SkinDiseaseDataset` class loads images and labels.
   - Augmentations include random flips, rotations, and normalization.
   - Datasets are split into Train/Validation/Test sets (70/20/10).

2. **Model Initialization**:
   - EfficientNet-B3 model initialized with ImageNet pre-trained weights.
   - Last classification layer modified to match the number of skin disease classes.

3. **Training Loop**:
   - Forward pass with automatic mixed precision.
   - Loss computation and backward pass.
   - Optimizer step and learning rate scheduler update.
   - Metrics computed: training and validation accuracy.

4. **Checkpointing**:
   - Best model (based on validation accuracy) is saved automatically during training.

5. **Logging**:
   - Training loss, validation loss, and accuracies are logged at each epoch.
   - Time taken per epoch is recorded for efficiency analysis.

---

## 📈 Key Metrics Monitored

- **Training Loss**
- **Validation Loss**
- **Training Accuracy**
- **Validation Accuracy**
- **Time per Epoch**
- **GPU Memory Usage**

---

## 📌 Notes

- Training uses **DataParallel** when multiple GPUs are available.
- Mixed precision training significantly reduces memory usage and improves speed.
- Early stopping can be manually incorporated if validation loss stops improving.
- Weighted loss handles class imbalance common in skin disease datasets.

---

## 🚀 Conclusion

This training pipeline allows efficient and scalable fine-tuning of large CNN models like **EfficientNet-B3** on imbalanced medical datasets.  
The use of augmentation, mixed precision, and weighted loss improves generalization and robustness of the trained model.

---
