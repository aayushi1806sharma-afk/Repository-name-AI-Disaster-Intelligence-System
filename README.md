# 🚨 AI-Powered Disaster Intelligence System

A computer vision system that classifies disaster-related images into 5 categories — **Fire, Flood, Landslide, Normal, and Smoke** — using transfer learning (MobileNetV2) and PyTorch. Includes a Streamlit web app for real-time predictions.

## 📊 Dataset

- **Source:** [Kaggle - Disaster Damage 5-Class Dataset](https://www.kaggle.com/datasets/sarthaktandulje/disaster-damage-5class)
- **Total Images:** 8,081
- **Classes:** Fire (2537), Flood (2706), Landslide (310), Normal (2226), Smoke (302)

## 🧠 Model

- **Architecture:** MobileNetV2 (pretrained on ImageNet) with a custom classifier head
- **Approach:** Transfer learning with frozen backbone — only the final classifier layer is trained
- **Why this approach:** With limited CPU resources and a class-imbalanced dataset (landslide/smoke have far fewer samples), training from scratch was impractical. Transfer learning leverages pretrained visual features, drastically reducing training time while maintaining strong accuracy.
- **Class Imbalance Handling:** Weighted Cross-Entropy Loss using `sklearn.compute_class_weight`

## 📈 Results

- **Overall Validation Accuracy:** 89%

| Class | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Fire | 0.92 | 0.96 | 0.94 |
| Flood | 0.92 | 0.92 | 0.92 |
| Landslide | 0.63 | 0.80 | 0.71 |
| Normal | 0.89 | 0.83 | 0.86 |
| Smoke | 0.85 | 0.80 | 0.82 |

### Training Curves
![Training Curves](reports/training_curves.png)

### Confusion Matrix
![Confusion Matrix](reports/confusion_matrix.png)

## 🖥️ Tech Stack

- Python, PyTorch, Torchvision
- NumPy, Pandas, Matplotlib, Seaborn
- OpenCV
- Streamlit
- scikit-learn

## 🚀 How to Run

1. Clone the repository: