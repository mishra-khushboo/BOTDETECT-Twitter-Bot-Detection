# 🤖 Twitter Bot Detection using Deep Learning & XAI

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-red?logo=pytorch)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A deep learning pipeline to detect Twitter bots using a combination of user metadata, account features, and TinyBERT-based tweet embeddings — with Explainable AI (LIME) to interpret model decisions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Models](#models)
- [Results](#results)
- [Explainable AI](#explainable-ai-xai)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Key Findings](#key-findings)

---

## Overview

Social media bots pose a significant threat to online discourse by spreading misinformation, manipulating trends, and impersonating real users. This project builds and evaluates three deep learning models — RNN, CNN, and DNN — to classify Twitter accounts as **bot** or **non-bot**, and uses LIME to explain individual predictions.

The pipeline covers the full ML lifecycle: raw data cleaning → feature engineering → TinyBERT embeddings → hyperparameter tuning with Optuna → model training → evaluation → explainability.

---

## Dataset

The dataset (`TwitterData.csv`) contains **279,691 tweets** across **15 columns**:

| Column | Description |
|---|---|
| `Twitter_User_Name` | Display name of the user |
| `Twitter_Account` | Account handle |
| `Twitter_User_Description` | Bio/description |
| `Tweet_text` | Raw tweet content |
| `Label` | 0 = Bot, 1 = Human (later inverted) |
| `Followers` / `Following` | Engagement counts |
| `Verified` | Account verification status |
| `Retweet` | Whether tweet is a retweet |
| `Location` / `Real_Location` | User location info |
| `Link` | Tweet link |

---

## Project Pipeline

```
Raw Data
   │
   ▼
Data Preprocessing (null handling, column dropping)
   │
   ▼
Text Cleaning (URLs, mentions, stopwords removed via NLTK)
   │
   ▼
User-based Train/Test Split (70/30, no data leakage)
   │
   ▼
Feature Engineering (username/account metadata features)
   │
   ▼
Normalisation (MinMaxScaler on continuous features)
   │
   ▼
TinyBERT Embeddings (312-dim vectors from combined tweet + bio text)
   │
   ▼
Feature Concatenation (tabular + embeddings = 324 features)
   │
   ▼
Model Training (RNN / CNN / DNN with Optuna tuning)
   │
   ▼
Evaluation (Accuracy, F1, Confusion Matrix)
   │
   ▼
XAI with LIME (per-instance explanations)
```

---

## Models

### 1. RNN (Simple Recurrent Neural Network)
- Architecture: `SimpleRNN → Dropout → Dense(64) → Dense(1, sigmoid)`
- Tuned with Optuna (10 trials on 5k-sample subset)
- Best params: units=48, dropout=0.237, lr=0.000258

### 2. CNN (1D Convolutional Neural Network)
- Architecture: `Conv1D → MaxPooling1D → Flatten → Dropout → Dense(64) → Dense(1, sigmoid)`
- Tuned with Optuna (3 trials)
- Best params: filters=48, kernel_size=3, dropout=0.20, lr=0.000123

### 3. DNN (Deep Neural Network) ⭐ Best Model
- Framework: PyTorch
- Architecture: `Linear → ReLU → Dropout → Linear(2) → Softmax`
- Tuned with Optuna (40 trials — most thorough search)
- Best params: hidden_size=311, dropout=0.342, lr=5.65e-5
- Optimizers tested: **AdamW** and **AdaBelief**
- Additional experiment: DNN + SMOTE + BatchNorm + SGD + OneCycleLR

---

## Results

| Model | Test Accuracy | Notes |
|---|---|---|
| RNN | 63.05% | Overfitting; sequence model not suited for flat feature vectors |
| CNN | 88.93% | Strong local pattern detection across embedding dimensions |
| **DNN (AdaBelief)** | **93.25%** | Best overall; well-tuned feedforward network |

### DNN Classification Report

```
              precision    recall  f1-score   support

     Non-Bot       0.94      0.90      0.92     37419
         Bot       0.92      0.96      0.94     44930

    accuracy                           0.93     82349
   macro avg       0.93      0.93      0.93     82349
weighted avg       0.93      0.93      0.93     82349
```

---

## Explainable AI (XAI)

[LIME](https://github.com/marcotcr/lime) (Local Interpretable Model-agnostic Explanations) was used to explain individual predictions from the best DNN model.

For each instance, LIME perturbs the input features, observes changes in model output, and fits a local linear model to estimate feature importance.

**Key signals identified by LIME:**
- Low follower/following counts → Bot indicator
- `Verified = 0` → Bot indicator
- `Contains_Bot = 1` in account name → Strong bot signal
- Embedding dimensions encoding bot-like language patterns → Influential features

Explanations were generated for multiple test instances and saved as interactive HTML files.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/mishra-khushboo/BOTDETECT-Twitter-Bot-Detection.git
cd BOTDETECT-Twitter-Bot-Detection
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

---

## Usage

Run each notebook stage in order:

```bash
# 1. Data Preprocessing & Cleaning
jupyter notebook notebooks/01_preprocessing.ipynb

# 2. Feature Engineering & Normalisation
jupyter notebook notebooks/02_feature_engineering.ipynb

# 3. TinyBERT Embeddings
jupyter notebook notebooks/03_embeddings.ipynb

# 4. Feature Concatenation
jupyter notebook notebooks/04_concatenation.ipynb

# 5. Model Training
jupyter notebook notebooks/05_rnn_model.ipynb
jupyter notebook notebooks/06_cnn_model.ipynb
jupyter notebook notebooks/07_dnn_model.ipynb

# 6. XAI & Evaluation
jupyter notebook notebooks/08_xai_lime.ipynb
```

---

## Project Structure

```
twitter-bot-detection/
│
├── data/
│   ├── TwitterData.csv               # Raw dataset
│   ├── TwitterData_Cleaned.csv       # After text cleaning
│   ├── train_data_normalized.csv     # Normalised train split
│   ├── test_data_normalized.csv      # Normalised test split
│   ├── train_embeddings.npy          # TinyBERT train embeddings
│   ├── test_embeddings.npy           # TinyBERT test embeddings
│   ├── train_data_final_modified.csv # Final train features + embeddings
│   └── test_data_final_modified.csv  # Final test features + embeddings
│
├── models/
│   ├── final_rnn_model.h5            # Saved RNN model
│   ├── final_cnn_model.h5            # Saved CNN model
│   ├── dnn_model_optimadamw.pth      # DNN with AdamW
│   └── dnn_model_optimadabelief.pth  # DNN with AdaBelief (best)
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_embeddings.ipynb
│   ├── 04_concatenation.ipynb
│   ├── 05_rnn_model.ipynb
│   ├── 06_cnn_model.ipynb
│   ├── 07_dnn_model.ipynb
│   └── 08_xai_lime.ipynb
│
├── outputs/
│   ├── lime_explanation.html         # LIME explanation for instance 0
│   ├── accuracy_chart.png            # Model comparison bar chart
│   └── train_vs_test_loss.png        # Loss curves
│
├── requirements.txt
└── README.md
```

---

## Dependencies

```
pandas
numpy
scikit-learn
torch
tensorflow
transformers
optuna
lime
nltk
seaborn
matplotlib
imbalanced-learn
adabelief-pytorch
```

Install all at once:
```bash
pip install -r requirements.txt
```

> **Note:** TinyBERT embeddings generation requires a GPU for reasonable speed. Google Colab with T4 GPU is recommended.

---

## Key Findings

- **User-based splitting** is critical — row-level splitting causes data leakage since the same user appears multiple times.
- **TinyBERT embeddings** (312 dimensions) combined with tabular features outperform either alone.
- **RNN is poorly suited** for this task — the 324-feature input is not a true sequence, so recurrent layers find no temporal structure to exploit.
- **CNN performs well** because Conv1D kernels detect local patterns across adjacent embedding dimensions.
- **DNN with AdaBelief** achieves the best result (93.25%) through thorough Optuna hyperparameter search across 40 trials.
- **LIME confirms** the model uses meaningful signals: follower count, verification status, bot-related keywords in account names, and semantic embedding patterns.

---

## 📊 Model Accuracy Comparison

```
RNN  ████████████░░░░░░░░  63%
CNN  █████████████████░░░  89%
DNN  ██████████████████░░  93%
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [TinyBERT](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D) by Huawei Noah's Ark Lab
- [LIME](https://github.com/marcotcr/lime) by Marco Tulio Ribeiro
- [Optuna](https://optuna.org/) for hyperparameter optimisation
- [AdaBelief Optimizer](https://github.com/juntang-zhuang/Adabelief-Optimizer) by Juntang Zhuang
