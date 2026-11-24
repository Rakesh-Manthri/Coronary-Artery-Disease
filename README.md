# CardioScan AI: End-to-End CAD Diagnostic & Staging System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Scikit--Learn-1.9%2B-orange.svg)](https://scikit-learn.org/)
[![Dashboard](https://img.shields.io/badge/Streamlit-1.67%2B-red.svg)](https://streamlit.io/)
[![Clinical Dataset](https://img.shields.io/badge/Dataset-UCI%20Cleveland-green.svg)](https://archive.ics.uci.edu/dataset/45/heart+disease)

An end-to-end, clinically calibrated machine learning system for **Coronary Artery Disease (CAD)** screening, risk stratification, and therapeutic care staging. Built on the benchmark **UCI Cleveland Clinic** cohort, this repository encompasses data preprocessing, exploratory data analysis, class imbalance remediation (SMOTE & class weighting), multi-model benchmarking, dual-stage production deployment, and an interactive **Streamlit** clinical diagnostic portal.

---

## 📋 Table of Contents
- [Executive Summary](#-executive-summary)
- [System Architecture (Dual-Stage Cascade)](#-system-architecture-dual-stage-cascade)
- [Clinical Cohort & Feature Schema](#-clinical-cohort--feature-schema)
- [Repository Structure](#-repository-structure)
- [Research & Modeling Pipeline](#-research--modeling-pipeline)
- [Performance & Clinical Safety Metrics](#-performance--clinical-safety-metrics)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
  - [1. Interactive Web Application](#1-interactive-web-application)
  - [2. CLI & Automated Self-Test](#2-cli--automated-self-test)
  - [3. Batch CSV Scoring](#3-batch-csv-scoring)
  - [4. Programmatic Python API](#4-programmatic-python-api)
  - [5. Research PDF Generation](#5-research-pdf-generation)
- [Technology Stack](#-technology-stack)
- [License & Citation](#-license--citation)

---

## 🩺 Executive Summary

Coronary Artery Disease (CAD) remains the leading contributor to global cardiovascular mortality. While standard non-invasive diagnostic workflows rely on binary classification (presence vs. absence of disease), real-world clinical decision-making requires **graded risk stratification** to distinguish between:
1. **Healthy Individuals (Tier 0)**: Non-urgent, lifestyle counseling; avoid unnecessary invasive coronary angiography.
2. **Mild/Moderate CAD (Tier 1)**: Single/double-vessel disease managed conservatively with guideline-directed medical therapy (statins, antiplatelets, ACE inhibitors).
3. **Severe/Critical CAD (Tier 2)**: Multi-vessel or left main disease requiring urgent catheterization and revascularization (PCI / CABG).

CardioScan AI bridges this gap with a **calibrated two-stage cascade architecture** that achieves **92.86% diagnostic sensitivity** in screening outpatients while effectively staging multi-vessel CAD cases for immediate intervention.

---

## 🏗 System Architecture (Dual-Stage Cascade)

```
                       [Raw Patient Clinical Inputs]
                                     │
                                     ▼
                   [Zero-Leakage Preprocessing Pipeline]
       (Median Numeric Imputation, Mode Categorical Imputation, OHE, MinMax Scaling)
                                     │
                                     ▼
                      [Stage 1: Screening Engine]
                  Calibrated Multi-Layer Perceptron (MLP)
                Operating Threshold: tau* = 0.391 (Youden's J)
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
         [Predicted Clean]                   [Predicted CAD]
       Tier 0: Healthy Cohort                        │
       (Lifestyle / Routine Care)                     ▼
                                          [Stage 2: Care Staging]
                                         Cost-Sensitive Random Forest
                                        (Multi-Class Balanced Weights)
                                                      │
                                           ┌──────────┴──────────┐
                                           ▼                     ▼
                                  [Tier 1: Medical]     [Tier 2: Surgical]
                                  1-2 Vessel CAD        Severe / Multi-Vessel CAD
                                  Outpatient Statins    Urgent Angiography &
                                  & Antiplatelets       Revascularization (PCI/CABG)
```

---

## 📊 Clinical Cohort & Feature Schema

The system is trained and validated on the **UCI Cleveland Heart Disease** cohort (303 patient records, 14 clinical attributes):

| Feature | Type | Description | Clinical Reference Range |
|---|---|---|---|
| `age` | Numeric | Patient age in years | 29 – 77 |
| `sex` | Categorical | Biological sex | `1` = Male, `0` = Female |
| `cp` | Categorical | Chest pain type | `1`: Typical Angina, `2`: Atypical Angina, `3`: Non-Anginal, `4`: Asymptomatic |
| `trestbps` | Numeric | Resting blood pressure | 94 – 200 mm Hg |
| `chol` | Numeric | Serum cholesterol | 126 – 564 mg/dL |
| `fbs` | Categorical | Fasting blood sugar > 120 mg/dL | `1` = True, `0` = False |
| `restecg` | Categorical | Resting electrocardiographic results | `0`: Normal, `1`: ST-T Wave Abnormality, `2`: Left Ventricular Hypertrophy |
| `thalach` | Numeric | Maximum heart rate achieved | 71 – 202 bpm |
| `exang` | Categorical | Exercise-induced angina | `1` = Yes, `0` = No |
| `oldpeak` | Numeric | ST depression induced by exercise | 0.0 – 6.2 mm |
| `slope` | Categorical | Peak exercise ST segment slope | `1`: Upsloping, `2`: Flat, `3`: Downsloping |
| `ca` | Numeric | Major vessels colored by fluoroscopy | 0 – 3 vessels |
| `thal` | Categorical | Thallium scintigraphy stress test | `3`: Normal, `6`: Fixed Defect, `7`: Reversible Defect |

---

## 📁 Repository Structure

```
AIML/
├── 2 tier/                                # Binary classification research track
│   ├── preprocessing.ipynb                # Data cleaning, EDA, zero-leakage splits
│   ├── baseline.ipynb                     # Default baseline models evaluation
│   └── hyperparameter_tuning.ipynb        # Hyperparameter search & cross-validation
├── 3 tier/                                # Multi-class clinical severity staging track
│   ├── preprocessing.ipynb                # 3-tier label mapping and feature alignment
│   ├── baseline.ipynb                     # Multi-class baseline evaluation
│   └── hyperparameter_tuning.ipynb        # Cost-sensitive multi-tier optimization
├── With smote/                            # Resampling research: SMOTE synthetic oversampling
│   ├── preprocessing.ipynb
│   ├── baseline.ipynb
│   └── hyperparameter_tuning.ipynb
├── without smote/                         # Resampling research: Native class weighting
│   ├── preprocessing.ipynb
│   ├── baseline.ipynb
│   └── hyperparameter_tuning.ipynb
├── datasets/                              # Benchmark clinical datasets
│   ├── heart+disease/                     # Cleveland, Hungarian, Long Beach, Switzerland
│   └── statlog+heart/                     # Statlog validation cohort
├── deployment/                            # Production deployment artifacts
│   ├── models/                            # Serialized production pipeline
│   │   ├── cad_preprocessor.joblib        # Scikit-learn transformer pipeline
│   │   ├── cad_stage1_screening.joblib    # Calibrated MLP screening neural network
│   │   ├── cad_stage2_staging.joblib      # Balanced Random Forest staging classifier
│   │   └── pipeline_metadata.json         # Operating thresholds & schema definitions
│   ├── phase4_deployment.ipynb            # End-to-end production pipeline training
│   ├── predict_cad.py                     # Self-contained CLI and inference engine
│   └── README.md                          # Phase 4 deployment specifications
├── app.py                                 # Interactive Streamlit clinical web app
├── generate_pdf_report.py                 # Automated research report PDF generator
├── Heart_Disease_Classification_Research_Report.pdf  # Generated clinical publication report
├── requirements.txt                       # Project dependencies
└── README.md                              # Main project documentation
```

---

## 🔬 Research & Modeling Pipeline

The modeling lifecycle followed four systematic phases:

1. **Exploratory Data Analysis & Preprocessing**:
   - Median imputation for missing continuous values (`ca`, `thal`) to prevent data leakage.
   - Categorical encoding with fixed feature columns aligned across train/test splits.
   - MinMax scaling to bound continuous features within `[0, 1]`.

2. **Class Imbalance Remediation**:
   - Compared synthetic generation (**SMOTE**) against **cost-sensitive learning** (`class_weight='balanced'`).
   - Cost-sensitive weighting demonstrated superior generalization on clinical holdout sets without hallucinating non-physiological patient distributions.

3. **Multi-Algorithm Benchmarking**:
   - Benchmarked Logistic Regression, Support Vector Classifiers (SVC), Random Forests, Extra Trees, Gradient Boosting, AdaBoost, and Multi-Layer Perceptrons (MLP).
   - Applied **GridSearchCV** with stratified 5-fold cross-validation optimizing for ROC-AUC and Recall.

4. **Production Cascade Formulation (Phase 4)**:
   - De-coupled the diagnostic problem into a **high-sensitivity screening filter** followed by an **in-depth therapeutic staging classifier**.
   - Tuned decision threshold $\tau^* = 0.391$ via Youden's $J$ statistic to heavily penalize false negatives in Stage 1.

---

## 📈 Performance & Clinical Safety Metrics

Evaluated on an independent, unseen holdout cohort ($N = 61$):

| Metric | Stage 1 (Screening MLP) | Stage 2 (Staging Random Forest) |
|---|---|---|
| **Diagnostic Sensitivity (Recall)** | **92.86%** (26 / 28 CAD cases captured) | — |
| **Specificity (Healthy Retention)** | **72.73%** | — |
| **ROC-AUC Score** | **0.902** | **0.864** |
| **Tier 2 (Surgical) Sensitivity** | — | **67.0%** capture of multi-vessel disease |
| **Zero-Leakage Guarantee** | 100% strictly enforced | 100% strictly enforced |

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+ (Tested up to Python 3.14)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/cardioscan-ai.git
cd cardioscan-ai
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

### 1. Interactive Web Application
Launch the production Streamlit diagnostic dashboard:
```bash
streamlit run app.py
```
**Features:**
- **Single Patient Assessment**: Input clinical vitals, obtain instant triage status, risk probability gauge, and recommended treatment protocols.
- **Batch CSV Triage**: Upload patient lists and export structured clinical recommendations.
- **Model Architecture & Safety**: Interactive cascade diagrams, calibration curves, and confusion matrices.
- **Synthetic Patient Presets**: Test clean, moderate, and critical clinical profiles with one click.

### 2. CLI & Automated Self-Test
Run the built-in self-test suite:
```bash
python deployment/predict_cad.py --test
```
Evaluate an individual patient via CLI flags:
```bash
python deployment/predict_cad.py --age 62 --sex 1 --cp 4 --trestbps 140 --chol 268 --fbs 0 --restecg 2 --thalach 160 --exang 0 --oldpeak 3.6 --slope 3 --ca 2 --thal 3
```

### 3. Batch CSV Scoring
Score a cohort from a CSV file:
```bash
python deployment/predict_cad.py --batch input_patients.csv --output scored_patients.csv
```

### 4. Programmatic Python API
```python
from deployment.predict_cad import CADInferenceEngine

engine = CADInferenceEngine()

patient = {
    'age': 58.0, 'sex': 1, 'cp': 4, 'trestbps': 136.0, 'chol': 250.0,
    'fbs': 0, 'restecg': 1, 'thalach': 140.0, 'exang': 1, 'oldpeak': 1.6,
    'slope': 2, 'ca': 1.0, 'thal': 7
}

result = engine.predict_patient(patient)
print(f"Status: {result['screening_status']}")
print(f"CAD Risk: {result['cad_probability']:.1%}")
print(f"Care Tier: {result['recommended_care_tier']} ({result['tier_description']})")
print(f"Action: {result['clinical_action_plan']}")
```

### 5. Research PDF Generation
Re-compile the formal clinical research report:
```bash
python generate_pdf_report.py
```
Outputs `Heart_Disease_Classification_Research_Report.pdf` with publication-grade typography, tables, and architecture schemas.

---

## 🛠 Technology Stack

- **Data Processing**: `numpy`, `pandas`, `scipy`
- **Machine Learning**: `scikit-learn`, `imbalanced-learn`, `joblib`
- **Visualization**: `matplotlib`, `seaborn`
- **Deployment & Web UI**: `streamlit`
- **Document Engine**: `reportlab`
- **Exploration Environment**: `jupyter`, `ipykernel`

---

## 📄 License & Citation

This project is developed for educational and research purposes under the **MIT License**. Clinical assessments produced by this tool must be corroborated by licensed medical practitioners and coronary angiographic imaging prior to interventional therapy.
