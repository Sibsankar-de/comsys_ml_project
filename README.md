# 🧠 Deep Learning Models for Gender Classification & Face Recognition
*Team:* Code_Slayers  
*Date:* June 29, 2025  

This document summarizes two deep learning-based models developed to perform *gender classification* and *face recognition* under visually challenging conditions such as blur, fog, rain, low light, and overexposure.

---

## 🧔‍♀ Gender Classification Model

### 🎯 Objective
To classify facial images into *male* or *female, even when the images are degraded by environmental noise or distortion. The focus is on model generalization **without noise-specific preprocessing*.

### 📂 Dataset
- Organized into train/ and val/ folders
- Each with male/ and female/ subfolders
- Contains raw and potentially noisy face images

### ⚙ Preprocessing
- Resizing and normalization (zero mean, unit variance)
- Data augmentation (random flips, rotations)
- Loaded using PyTorch's ImageFolder and DataLoader

### 🧠 Model Architecture
- CNN with:
  - Convolutional + ReLU layers
  - MaxPooling
  - Fully Connected layers
  - Dropout
  - Sigmoid output for binary classification
- *Loss Function:* BCELoss  
- *Optimizer:* Adam  
- GPU-compatible training

### 📊 Results
- *Accuracy:* 91%  
- *F1-score (Male):* 0.95  
- *F1-score (Female):* 0.74  
- Class imbalance led to lower performance on female class

### 💡 Key Contributions
- Strong performance in real-world image conditions
- No preprocessing for environmental noise
- Highlights fairness challenges in imbalanced datasets

### 🔮 Future Work
- Balance dataset
- Improve fairness across genders
- Use synthetic noise-aware data for augmentation

---

## 🧑‍💼 Face Recognition Model

### 🎯 Objective
To recognize *individual identities* in face images affected by distortions like fog, blur, rain, and overexposure. The model aims for *robust identity classification* under real-world visual degradation.

### 📂 Dataset
- Structured into train/ and val/ folders
- Each identity folder (001frontal, 002frontal, etc.) contains:
  - Clean frontal face image
  - A distortion/ folder with noisy images of the same person

### ⚙ Preprocessing
- Resized to 224×224
- Normalized (mean, std)
- Augmented with flips and rotations
- Custom Dataset class used for image-label mapping

### 🧠 Model Architecture
- ResNet-inspired CNN:
  - Convolution + BatchNorm + ReLU layers
  - MaxPooling
  - Fully Connected layers for embedding
  - Final classification layer (identity)
- *Loss Function:* CrossEntropyLoss  
- *Optimizer:* Adam  
- GPU-accelerated training

### 📊 Results
- Evaluated using:
  - Top-1 accuracy
  - Confusion matrix
  - t-SNE / PCA visualization
- Strong recognition on most distortions
- Reduced accuracy under extreme overexposure or occlusion

### 💡 Key Contributions
- Distortion-invariant identity recognition
- Effective use of paired clean-distorted data
- Demonstrated resilience without distortion-specific tuning

### 🔮 Future Work
- Balance number of distorted samples per identity
- Explore contrastive/triplet loss for better embeddings
- Improve performance on extreme visual noise

---

## 🧩 Conclusion
Both models showcase the power of deep learning in handling real-world visual challenges. The *gender classification model* is suitable for semantic attribute prediction, while the *face recognition model* excels in identity verification. Future enhancements should focus on fairness, distortion balancing, and more robust feature learning.

---