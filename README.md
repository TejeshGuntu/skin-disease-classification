# 🧬 Skin Disease Classification using CNN, Transfer Learning & Explainable AI

An end-to-end deep learning project for **multi-class skin disease image classification** using Convolutional Neural Networks (CNNs), Transfer Learning with **ResNet18**, and Explainable AI using **Grad-CAM**.

The project covers the complete deep learning workflow, including dataset exploration, image preprocessing, baseline CNN development, transfer learning, fine-tuning, model evaluation, Grad-CAM-based interpretability, and deployment through Streamlit.

> ⚠️ **Disclaimer:** This project is intended for educational and research purposes only. It is not a substitute for professional medical diagnosis.

---

## 📌 Project Overview

Skin diseases can exhibit visually similar characteristics, making automated image classification a challenging computer vision problem.

This project uses deep learning to classify skin images into **eight disease categories**. A custom CNN is first used as a baseline, followed by transfer learning using a pre-trained **ResNet18** model.

To improve model transparency, **Gradient-weighted Class Activation Mapping (Grad-CAM)** is used to visualize the image regions that contributed most to the model's prediction.

### Project Pipeline

Skin Image  
↓  
Image Preprocessing & Normalization  
↓  
Baseline CNN  
↓  
ResNet18 Transfer Learning  
↓  
Fine-Tuning  
↓  
8-Class Disease Prediction  
↓  
Confidence Score  
↓  
Grad-CAM Visualization

---

## ✨ Key Highlights

- 🖼️ Multi-class classification of **8 skin disease categories**
- 🧠 Baseline **Custom CNN**
- 🔄 Transfer Learning using **ResNet18**
- 🎯 Fine-tuning of the pre-trained model
- ⚖️ Class imbalance handling using **Weighted CrossEntropyLoss**
- 📊 Model evaluation using Accuracy, Precision, Recall, F1-score, and Confusion Matrix
- 🔍 Explainable AI using **Grad-CAM**
- 🚨 Confidence-based prediction threshold for low-confidence predictions
- 🌐 Interactive deployment using **Streamlit**

---

## 📸 Application Demo

| Application Interface | Prediction Interface |
|---|---|
| ![Demo 1](assets/Demo1.png) | ![Demo 2](assets/Demo2.png) |

---

# 📂 Dataset

This project uses the **Skin Disease Dataset** available on Kaggle:

[Skin Disease Dataset on Kaggle](https://www.kaggle.com/datasets/subirbiswas19/skin-disease-dataset)

The dataset is organized into training and testing directories containing eight skin disease classes.

### Dataset Structure

Dataset/
├── train_set/
│   ├── BA-cellulitis/
│   ├── BA-impetigo/
│   ├── FU-athlete-foot/
│   ├── FU-nail-fungus/
│   ├── FU-ringworm/
│   ├── PA-cutaneous-larva-migrans/
│   ├── VI-chickenpox/
│   └── VI-shingles/
│
└── test_set/
    ├── BA-cellulitis/
    ├── BA-impetigo/
    ├── FU-athlete-foot/
    ├── FU-nail-fungus/
    ├── FU-ringworm/
    ├── PA-cutaneous-larva-migrans/
    ├── VI-chickenpox/
    └── VI-shingles/

---

# 🩺 Skin Disease Classes

The model classifies images into the following categories:

| Class | Disease Category |
|---|---|
| 1 | BA-cellulitis |
| 2 | BA-impetigo |
| 3 | FU-athlete-foot |
| 4 | FU-nail-fungus |
| 5 | FU-ringworm |
| 6 | PA-cutaneous-larva-migrans |
| 7 | VI-chickenpox |
| 8 | VI-shingles |

---

# 🔍 Exploratory Data Analysis

The dataset was inspected before model development to understand:

- Distribution of images across disease classes
- Image dimensions and quality
- Class imbalance
- Sample images from each category

Medical image datasets can contain variations in:

- Lighting conditions
- Backgrounds
- Skin appearance
- Image quality
- Lesion size and location

These variations make robust preprocessing and feature extraction important for model performance.

---

# 🧠 Model Development

## 1️⃣ Baseline Model — Custom CNN

A custom Convolutional Neural Network was developed as a baseline to establish an initial performance benchmark.

The CNN learns hierarchical image features through:

Input Image  
↓  
Convolution Layers  
↓  
Activation Functions  
↓  
Pooling Layers  
↓  
Feature Extraction  
↓  
Fully Connected Layers  
↓  
8-Class Classification

### Baseline Performance

**Test Accuracy: 67.09%**

The baseline model was used as a reference for evaluating the benefits of transfer learning.

---

# 2️⃣ Transfer Learning using ResNet18

A pre-trained **ResNet18** model was used to improve image feature extraction.

ResNet18 was initialized with ImageNet pre-trained weights and adapted for the eight-class skin disease classification task.

### Model Architecture

Input Image (224 × 224)  
↓  
Pre-trained ResNet18 Backbone  
↓  
Feature Extraction  
↓  
Modified Fully Connected Layer  
↓  
8-Class Output

The final classification layer was modified to match the number of disease classes.

---

# ⚖️ Handling Class Imbalance

To reduce potential bias toward classes with more training examples, class weights were incorporated into the loss function.

### Loss Function

**Weighted CrossEntropyLoss**

Class weighting increases the importance of underrepresented classes during training and helps the model optimize performance across all disease categories.

---

# 📈 Training Strategy

## Phase 1 — Frozen Backbone

During the first transfer learning phase:

- Pre-trained ResNet18 feature layers were frozen
- Only the final classifier layer was trained
- Training focused on adapting the model to the skin disease dataset

### Results

| Metric | Score |
|---|---:|
| Accuracy | **88.03%** |
| Macro Recall | **88.03%** |
| Macro F1-Score | **86.81%** |

This demonstrated a significant improvement compared with the custom CNN baseline.

---

# 🔥 Phase 2 — Fine-Tuning

To further improve performance:

- Selected ResNet18 layers were unfrozen
- The model was fine-tuned on the skin disease dataset
- A learning-rate scheduling strategy was applied during training

Fine-tuning allows higher-level convolutional layers to adapt their learned ImageNet features to skin disease patterns.

---

# 🏆 Final Model Performance

The final fine-tuned ResNet18 model achieved:

| Metric | Score |
|---|---:|
| Test Accuracy | **95.73%** |
| Macro Recall | **95%** |
| Macro F1-Score | **95%** |

### Performance Improvement

**Custom CNN: 67.09%**  
↓  
**ResNet18 Transfer Learning: 88.03%**  
↓  
**Fine-Tuned ResNet18: 95.73%**

---

# 📊 Model Evaluation

The project evaluates the model using multiple metrics rather than accuracy alone.

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Macro-Averaged Metrics
- Confusion Matrix

### Visualizations

The project includes visualizations for:

- Training and validation performance
- Loss curves
- Accuracy metrics
- Recall metrics
- F1-score
- Confusion Matrix
- Sample predictions

---

# 🔍 Explainable AI using Grad-CAM

Deep learning models are often difficult to interpret.

To provide visual explanations for model predictions, this project implements:

> **Gradient-weighted Class Activation Mapping (Grad-CAM)**

Grad-CAM highlights regions of the input image that contributed most strongly to the predicted class.

---

## 🧠 How Grad-CAM Works

Input Image  
↓  
Forward Pass through ResNet18  
↓  
Predicted Class  
↓  
Compute Gradients  
↓  
Calculate Importance of Feature Maps  
↓  
Generate Class Activation Map  
↓  
Overlay Heatmap on Original Image

The Grad-CAM implementation captures:

- Activations from the target convolutional layer
- Gradients associated with the predicted class

The resulting activation map is converted into a heatmap and overlaid on the original skin image.

### Target Layer

**ResNet18 → layer4[1].conv2**

---

# 🔎 Grad-CAM Insights

Grad-CAM helps analyze which image regions contributed most to a model prediction.

For correctly classified images, the heatmaps can be used to examine whether the model focuses on relevant lesion regions rather than irrelevant background features.

For misclassified images, Grad-CAM provides additional insight into:

- Which visual regions influenced the prediction
- Potential similarity between disease classes
- Possible model weaknesses

> Grad-CAM improves interpretability but should not be considered proof of clinical reasoning or diagnostic reliability.

---

# 🚨 Confidence-Based Prediction Threshold

The Streamlit application uses a confidence threshold of:

**70%**

The prediction workflow is:

Image  
↓  
ResNet18 Prediction  
↓  
Softmax Probabilities  
↓  
Maximum Confidence Score  
↓  
Confidence ≥ 70%?

**YES → Display Prediction**  
**NO → Warn About Low Confidence**

This mechanism helps prevent the application from displaying predictions when the model confidence is below the defined threshold.

---

# 🌐 Streamlit Deployment

The trained model is deployed as an interactive Streamlit application.

### Application Features

- 📤 Upload skin images
- 🧠 AI-based disease classification
- 📊 Prediction confidence score
- 🔍 Grad-CAM heatmap visualization
- ⚠️ Educational-use disclaimer

### Supported Image Formats

- `.jpg`
- `.jpeg`
- `.png`

---

# 🛠️ Tech Stack

### Programming & Deep Learning

- Python
- PyTorch
- Torchvision

### Deep Learning Models

- Custom CNN
- ResNet18

### Computer Vision

- OpenCV
- PIL

### Data Processing

- NumPy

### Visualization

- Matplotlib

### Deployment

- Streamlit

---

# 📁 Repository Structure

skin-disease-classification/
│
├── Restnet30_finetuned_skin_dataset.ipynb
├── app.py
├── main.py
├── fine_tuned_skin_disease_model_unfrozen.pth
├── requirements.txt
├── pyproject.toml
│
├── assets/
│   ├── Demo1.png
│   └── Demo2.png
│
└── README.md

---
# 🚀 Installation

## 1. Clone the Repository

    git clone <your-repository-url>
    cd skin-disease-classification

## 2. Install Dependencies

    pip install -r requirements.txt

## 3. Run the Streamlit Application

    streamlit run app.py

---

# 🧪 Using the Application

1. Launch the Streamlit application.
2. Upload a skin image.
3. The image is resized to **224 × 224**.
4. The image is normalized using ImageNet normalization values.
5. ResNet18 predicts one of the eight classes.
6. The confidence score is calculated.
7. If the prediction meets the confidence threshold, the predicted class is displayed.
8. Grad-CAM generates a visualization showing important image regions.

---

# ⚠️ Limitations

This project has several important limitations:

- The model is trained only on the eight disease categories included in the dataset.
- Performance on external clinical datasets has not been established.
- High test accuracy does not guarantee real-world clinical performance.
- Image quality and dataset characteristics can influence predictions.
- Grad-CAM provides interpretability but does not provide clinical validation.
- The application should not be used as a replacement for professional medical diagnosis.

---

# 🔮 Future Improvements

Potential improvements include:

- Training on larger and more diverse dermatology datasets
- External dataset validation
- Advanced image augmentation strategies
- Comparison with EfficientNet and Vision Transformers
- Model calibration and uncertainty estimation
- Improved confidence-based rejection mechanisms
- Systematic fairness and bias evaluation
- Integration with additional explainability techniques
- Cloud deployment

---

# 🎯 Key Learning Outcomes

This project demonstrates practical experience with:

- Convolutional Neural Networks
- Transfer Learning
- Fine-Tuning Deep Learning Models
- Multi-Class Image Classification
- Medical Image Analysis
- Class Imbalance Handling
- Model Evaluation
- Explainable AI
- Grad-CAM
- PyTorch
- Streamlit Deployment

---
## 📦 Trained Model

The trained model weights (`.pth` file) are not included in this repository due to GitHub file size limitations.

The model can be reproduced by running the training notebook:

`Restnet30_finetuned_skin_dataset.ipynb`

# ⚠️ Disclaimer

This application is designed strictly for:

> **Educational, research, and experimental purposes.**

It is **not a medical device** and should not be used for medical diagnosis or treatment decisions.

Users should always consult qualified healthcare professionals for medical advice.
