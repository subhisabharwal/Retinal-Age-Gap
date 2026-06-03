# Retinal Age Gap Prediction using RETFound and Explainable Vascular Biomarker Analysis

## Overview

This project presents an explainable Artificial Intelligence (AI) framework for predicting retinal biological age from retinal fundus images using deep learning and vascular biomarker analysis.

The framework combines:
- Vision Transformer (ViT)-based retinal representation learning
- Retinal vessel segmentation
- Vascular biomarker extraction
- SHAP-based explainability

The primary objective is to estimate retinal biological age and compute the **Retinal Age Gap (RAG)**:

```math
Retinal Age Gap = Predicted Retinal Age − Chronological Age
```

The project explores multiple deep learning architectures including EfficientNet CNNs and RETFound Vision Transformer models for retinal age prediction.

---

# Problem Statement

Retinal fundus imaging provides a non-invasive view of vascular structures associated with biological aging and systemic health conditions.

Traditional retinal age prediction systems often lack interpretability, making it difficult to understand which retinal vascular patterns influence predictions.

This project aims to:
- Predict retinal biological age from fundus images
- Analyze retinal vascular biomarkers
- Improve interpretability using explainable AI
- Compare CNN and Vision Transformer architectures

---

# Objectives

- Develop an AI-based retinal age prediction framework
- Compare CNN and Vision Transformer models
- Extract retinal vascular biomarkers
- Perform explainability analysis using SHAP
- Compute Retinal Age Gap (RAG)
- Build an interpretable retinal aging analysis pipeline

---

# Dataset

## BRSET Dataset

The project uses retinal fundus images and associated clinical metadata from the BRSET retinal dataset.

### Dataset Components
- Retinal fundus images
- Patient age metadata
- Clinical attributes

---

# Dataset Preprocessing Pipeline

The preprocessing pipeline includes:

1. Image-Metadata Mapping
2. Missing Value Handling
3. Low Quality Image Removal
4. Image Standardization
5. Label Cleaning and Normalization
6. Data Augmentation
7. Train / Validation / Test Splitting

---

# Methodology

## Overall Pipeline

The proposed framework follows this workflow:

1. Dataset preprocessing
2. Retinal age prediction using deep learning models
3. Vessel segmentation using image enhancement techniques
4. Biomarker extraction using skeletonization and morphological analysis
5. XGBoost-based biomarker analysis
6. SHAP explainability analysis
7. Retinal Age Gap interpretation

---

# Deep Learning Models Explored

## 1. EfficientNet-B0

### Description
EfficientNet-B0 is a lightweight CNN-based baseline architecture used for retinal age prediction.

### Advantages
- Lightweight architecture
- Efficient feature extraction
- Faster training

### Limitation
- Limited ability to capture global retinal relationships

### Performance
- MAE: **8.29**

---

## 2. EfficientNet-B3

### Description
EfficientNet-B3 is a deeper CNN architecture with improved retinal feature representation capabilities.

### Advantages
- Better retinal texture learning
- Improved retinal feature representation

### Limitation
- Still limited in modeling long-range retinal dependencies

### Performance
- MAE: **7.73**

---

## 3. RETFound Linear Probe

### Description
RETFound is a Vision Transformer (ViT)-based retinal foundation model pretrained on retinal images.

In the linear probe strategy:
- RETFound encoder remained frozen
- Only the final regression head was trained

### Advantages
- Preserves pretrained retinal representations
- Reduces overfitting on smaller datasets
- Efficient transfer learning strategy

### Performance
- MAE: **6.77**
- Best numerical performance

---

## 4. RETFound Fine-Tuned Model

### Description
The fine-tuned RETFound model updates transformer weights for retinal age prediction.

### Advantages
- Captures long-range retinal relationships
- Learns contextual retinal representations
- Better understanding of vascular structures

### Performance
- MAE: **6.99**

---

# Model Comparison

| Model | Architecture | MAE | Key Strength |
|---|---|---|---|
| EfficientNet-B0 | CNN | 8.29 | Local retinal feature learning |
| EfficientNet-B3 | CNN | 7.73 | Improved retinal feature representation |
| RETFound Linear Probe | Vision Transformer (ViT) | 6.77 | Best numerical performance |
| RETFound Fine-Tuned | Vision Transformer (ViT) | 6.99 | Better contextual retinal understanding |

---

# Vessel Segmentation Pipeline

The retinal vessel extraction pipeline includes:

1. CLAHE-based contrast enhancement
2. Frangi filtering for vessel enhancement
3. Binary vessel mask generation
4. Skeletonization of vessel structures

---

# Vascular Biomarker Extraction

Quantitative vascular biomarkers were extracted from segmented retinal vessels using skeletonization and morphological analysis.

## Extracted Features

| Feature | Meaning |
|---|---|
| Vessel Density | Vessel coverage |
| Vessel Length | Total vessel span |
| Branching Points | Vascular complexity |
| Fractal Dimension | Structural irregularity |
| Tortuosity | Vessel curvature |
| Avg Thickness | Vessel caliber |

---

# Explainability using SHAP

SHAP (SHapley Additive exPlanations) was used to analyze the contribution of vascular biomarkers toward retinal age prediction.

## Purpose of SHAP

- Improves interpretability
- Quantifies feature importance
- Explains vascular aging indicators

## Example Insights

- Higher tortuosity increased retinal age prediction
- Reduced vessel density indicated vascular aging patterns

---

# Retinal Age Gap (RAG)

## Formula

```math
Retinal Age Gap = Predicted Retinal Age − Chronological Age
```

## Interpretation

- Positive RAG → accelerated retinal aging
- Negative RAG → healthier retinal aging

---

# Results

Key findings from the project:

- RETFound-based architectures outperformed CNN-based approaches
- Linear probing achieved the best numerical MAE
- SHAP improved biomarker-level interpretability
- Retinal vascular biomarkers showed meaningful association with aging

---

# Challenges Faced

## 1. Dataset Quality Variability
- Inconsistent retinal image quality
- Noise and illumination variations

### Solution
- Image preprocessing and enhancement

---

## 2. Limited Computational Resources
- Vision Transformer models required high GPU memory

### Solution
- Linear probing strategy
- Efficient training configurations

---

## 3. Segmentation Noise
- Small vessels were difficult to isolate accurately

### Solution
- CLAHE enhancement
- Frangi vessel filtering
- Skeletonization refinement

---

# Deployability

The proposed framework can be deployed as an AI-assisted retinal screening and retinal aging analysis system.

## Potential Deployment Areas
- Hospitals
- Ophthalmology clinics
- Diagnostic centers
- Preventive healthcare systems
- Research laboratories

## Potential Applications
- Early vascular aging assessment
- Cardiovascular risk analysis
- Neurodegenerative disease screening
- Automated retinal health monitoring
- AI-assisted clinical decision support

---

# Repository Structure

```bash
Retinal-Age-Gap/
│
├── Notebooks/
│
├── src/
│   ├── augmentation.py
│   ├── data_loader.py
│   ├── dataset.py
│   ├── demo_pipeline.py
│   ├── evaluate.py
│   ├── extract_vascular_features.py
│   ├── gradcam.py
│   ├── model_efficient_b3.py
│   ├── model_retfound.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── shap_analysis.py
│   ├── train_effnet_b3.py
│   ├── train_retfound.py
│   ├── train_retfound_final.py
│   └── vessel_segmentation.py
│
├── sample_outputs/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Repository Note

Large datasets, pretrained model weights, logs, and generated outputs are not included in this repository due to storage limitations.

---

# Sample Outputs

The repository includes sample retinal vessel segmentation outputs, prediction plots, and explainability visualizations for demonstration purposes.

---

# Technologies Used

## Programming Language
- Python

## Libraries and Frameworks
- PyTorch
- OpenCV
- NumPy
- Pandas
- Scikit-image
- Matplotlib
- SHAP
- XGBoost

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/Retinal-Age-Gap.git
cd Retinal-Age-Gap
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Future Work

- Multi-center retinal datasets
- Attention-based explainability
- Integration of patient clinical metadata
- Real-time deployment systems
- Clinician-guided validation studies

---

# Conclusion

This project demonstrates an explainable AI framework for retinal age prediction using retinal fundus images and vascular biomarker analysis.

The proposed pipeline combines:
- Deep learning
- Vessel segmentation
- Biomarker extraction
- Explainable AI techniques

to provide interpretable retinal aging analysis.

RETFound-based Vision Transformer architectures demonstrated superior retinal representation learning compared to CNN-based approaches, while SHAP analysis improved understanding of biomarker contributions toward retinal aging predictions.

The framework highlights the potential of AI-assisted retinal imaging systems for future healthcare and clinical decision support applications.

---

# Authors

Retinal Age Gap Prediction Project

---

# License

This project is intended for academic and research purposes.
