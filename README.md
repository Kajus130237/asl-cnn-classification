# 🤟 ASL Hand Gesture Classification using CNN

> **Research project** — University of Aveiro, Erasmus exchange (2025–2026)  
> Course: Artificial Intelligence Applications | Instructor: Petia Georgieva

Classifying **27 American Sign Language (ASL) gestures** (A–Z + space) from RGB images using a compact Convolutional Neural Network.  
Achieved **92.00% test accuracy** without any external pose estimation (MediaPipe-free).

---
### DATA SET:
https://www.kaggle.com/datasets/srisahithis/american-sign-language-a-z-dataset-hand-landmarks/data
---

## 📊 Results at a Glance

| Metric | Score |
|--------|-------|
| Test Accuracy | **92.00%** |
| Weighted Precision | 93.69% |
| Weighted Recall | 92.00% |
| F1-Score | 0.9118 |
| Parameters | ~150,000 |

**vs. Baseline (HOG + SVM): 71.2%** — CNN outperforms by +20 percentage points.

---

## 🛠️ Technologies

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

---

## 🧠 Model Architecture

A lightweight 3-block CNN (~150K parameters):

```
Input (64×64×3 RGB)
    │
Conv2D(32) + ReLU → MaxPool → Dropout(0.3)
    │
Conv2D(64) + ReLU → MaxPool → Dropout(0.3)
    │
Conv2D(128) + ReLU → GlobalAvgPool → Dropout(0.3)
    │
Dense(27, Softmax)
```

---

## 🔬 Methodology

### Dataset
- **9,599 images** across 27 classes (A–Z + space)
- Training: 6,750 | Validation: 1,349 | Test: 675 (balanced, 25/class)

### Data Augmentation (training only)
- Random rotation ±30°, translation ±25%, shear 25%
- Zoom 30%, brightness 0.8–1.2, horizontal flip

### Hyperparameter Optimization (3-stage grid search)
| Parameter | Values Tested | Best |
|-----------|--------------|------|
| Learning rate | 0.002, 0.001, 0.0005 | **0.002** |
| Filter config | various | **(32, 64, 128)** |
| Dropout | 0.1, 0.3, 0.5 | **0.3** |

---

## 📈 Per-class Highlights

**Perfect classification (F1 = 1.00):** D, H, I, L, Q

**Most challenging confusions:**
- B ↔ F (visually similar finger positions)
- R ↔ U / V (subtle orientation differences)

22 out of 27 classes achieved F1-score ≥ 0.90.

---

## 🚀 How to Run

```bash
git clone https://github.com/YOUR_USERNAME/asl-cnn-classification.git
cd asl-cnn-classification
pip install -r requirements.txt
jupyter notebook train.ipynb
```

**Dataset:** [ASL A–Z Dataset with Hand Landmarks — Kaggle](https://www.kaggle.com/)

---

## 📄 Full Report

The complete research paper with methodology, results, confusion matrix, and analysis is available here:  
📎 [`ASL_Hand_gesture_classification_CNN.pdf`](./ASL_Hand_gesture_classification_CNN.pdf)

---

## 👥 Authors

**Kajus Misevičius** — Dataset preparation, CNN implementation & training, confusion matrix analysis, report writing  

📧 kajus.miseviciu@ua.pt  
