# Deep Learning for Gender Classification & Face Recognition

**Team:** Code_Slayers  
**Date:** June 29, 2025

This repository contains two deep learning models developed using PyTorch to solve:

- **Task A:** Gender classification  
- **Task B:** Face recognition

Both models are trained and evaluated under visually challenging conditions such as blur, rain, fog, low light, and overexposure. Pretrained models are provided for quick testing and evaluation.

---

## 📁 Project Structure

```
.
├── gender_classification.ipynb        # Notebook for Task A
├── face_recognition.ipynb             # Notebook for Task B
├── best_gender_model.pth              # Trained model for gender classification
├── best_face_model.pth                # Trained model for face recognition
└── data-set/
    ├── Task_A/
    │   ├── train/
    │   │   ├── male/
    │   │   └── female/
    │   └── val/
    │       ├── male/
    │       └── female/
    └── Task_B/
        ├── train/
        │   ├── frontal_0001/
        │   │   ├── image.jpg
        │   │   └── distortion/
        │   │       ├── noisy1.jpg
        │   │       └── ...
        └── val/
            ├── frontal_0001/
            │   ├── image.jpg
            │   └── distortion/
            └── ...
```

---

## Task A: Gender Classification

### 🎯 Objective

Classify facial images into **male** or **female**, even when distorted by environmental noise or image degradation.

### 🧠 Model Architecture

- Pretrained CNN (e.g., ResNet) as feature extractor  
- Convolutional layers with ReLU  
- MaxPooling  
- Fully Connected layers  
- Dropout for regularization  
- Final Sigmoid activation for binary output  
- **Loss Function:** Binary Cross-Entropy (BCELoss)  
- **Optimizer:** Adam

### 🧪 Evaluation

- **Accuracy:** 91%  
- **F1-score (Male):** 0.95  
- **F1-score (Female):** 0.74 (due to class imbalance)

### ▶️ How to Run

1. **Install dependencies:**

```
pip install torch torchvision matplotlib seaborn scikit-learn tqdm
```

2. **Run the notebook:**

Open and execute all cells in `gender_classification.ipynb`.  
It will:
- Load the dataset from `data-set/Task_A/`
- Load the pretrained model `best_gender_model.pth`  
- Display classification report and confusion matrix

---

## Task B: Face Recognition

### 🎯 Objective

Recognize individual identities even under image distortions like fog, blur, and overexposure.

### 🧠 Model Architecture

- ResNet-inspired custom CNN  
- Convolution → BatchNorm → ReLU → MaxPooling  
- Fully connected embedding layer  
- Final classification layer with Softmax  
- **Loss Function:** Cross-Entropy Loss  
- **Optimizer:** Adam

### 🧪 Evaluation

- Evaluated using Top-1 Accuracy and Confusion Matrix  
- Robust under most distortions, except extreme overexposure

### ▶️ How to Run

1. **Install dependencies:**

```
pip install torch torchvision matplotlib seaborn scikit-learn
```

2. **Run the notebook:**

Open and execute all cells in `face_recognition.ipynb`.  
It will:
- Load the dataset from `data-set/Task_B/`
- Load the pretrained model `best_face_model.pth`  
- Display classification results and confusion matrix

---

## 🧪 Common Evaluation Tools

- **Classification Report** (Precision, Recall, F1-score)  
- **Confusion Matrix** (Seaborn heatmap)  
- **t-SNE / PCA** (optional for visualization of embeddings in face recognition)

---

## 🔮 Future Improvements

- Balance dataset for fairer gender classification  
- Use Triplet or Contrastive Loss for face embeddings  
- Introduce synthetic noise for better generalization

---

## ⚙ Environment Requirements

- Python 3.8+  
- PyTorch >= 1.10  
- torchvision  
- scikit-learn  
- seaborn  
- matplotlib  
- tqdm  
- Jupyter Notebook


---

## 🚀 GPU Acceleration

Both models are trained and evaluated using **GPU acceleration with CUDA** when available.

- The notebooks automatically move the model and tensors to the GPU (`cuda`) if available.
- This significantly reduces training time and improves evaluation speed.
- Make sure your system has an NVIDIA GPU with CUDA installed, or use platforms like Google Colab for GPU access.
