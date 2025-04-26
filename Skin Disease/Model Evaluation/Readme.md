# Model Evaluation for Skin Disease Classification

This notebook evaluates the performance of a trained **EfficientNet-B3** model for multi-class skin disease classification.  
The evaluation focuses on accuracy metrics, confusion matrix analysis, and visualizations of prediction performance.

---

## 📋 System and Model Information

- **Framework**: PyTorch
- **Model**: EfficientNet-B3 (Pretrained backbone)
- **Dataset**: Skin Disease Images Dataset (Custom curated)
- **Evaluation Metrics**:
  - Overall Accuracy
  - Class-wise Accuracy
  - Confusion Matrix
  - Visual Prediction Samples

---

## 🧪 Evaluation Steps

1. **Loading Trained Model**:
   - Loads the trained model weights.
   - Switches the model to `eval()` mode to disable dropout and batch norm updates.

2. **Data Preparation**:
   - Loads the **test dataset** split.
   - Applies necessary transformations (resizing, normalization).

3. **Prediction and Metrics Calculation**:
   - Performs forward pass on the test set.
   - Calculates:
     - **Overall Test Accuracy**
     - **Per-Class Accuracy**
     - **Confusion Matrix**

4. **Visualization**:
   - Plots a **confusion matrix heatmap** to visualize classification performance.
   - Displays **sample predictions**: showing actual vs. predicted class for test images.

---

## 📈 Key Results

- **Test Set Accuracy**: Achieved high overall accuracy indicating good generalization.
- **Per-Class Analysis**:
  - Some classes had near-perfect prediction rates.
  - Misclassifications were observed primarily between visually similar diseases.
- **Confusion Matrix Insights**:
  - Strong diagonal dominance indicating correct classifications.
  - Off-diagonal elements highlight common misclassification pairs.

---

## 📊 Visualizations

- **Confusion Matrix**:
  - Helps identify specific class-level weaknesses.
  - Warmer colors (dark red) indicate higher classification confidence.
- **Sample Predictions**:
  - Visual check on model behavior on unseen images.
  - Displays true vs predicted class names.

---

## 📌 Notes

- The model evaluation strictly uses the **test set** which was not seen during training or validation.
- **Mixed Precision** (AMP) was not used during inference to prioritize numerical stability.
- Confusion matrix was plotted using **Seaborn** and **Matplotlib** libraries.

---

## 📂 Output Files

- Evaluation plots and sample prediction images are saved locally for report generation and analysis.
- Confusion matrix visualization saved as:  
  `outputs/confusion_matrix.png`
- Sample prediction grid saved as:  
  `outputs/sample_predictions.png`

---

## 📷 Example Visualizations

> **Confusion Matrix**
>
> ![Confusion Matrix Example](./outputs/confusion_matrix.png)

> **Sample Predictions**
>
> ![Sample Predictions Example](./outputs/sample_predictions.png)

---

## 🚀 Conclusion

The evaluation confirms that **EfficientNet-B3**, fine-tuned on the skin disease dataset, achieves strong performance across diverse skin condition categories.  
The confusion matrix and per-class analysis offer insights for future model improvements, such as addressing misclassifications between visually similar diseases.

---
