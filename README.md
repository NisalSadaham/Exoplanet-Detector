# Exoplanet Detector

A Machine Learning project that classifies Kepler Objects of Interest (KOIs) as confirmed exoplanets or false positives using Logistic Regression.

## Project Overview

The Exoplanet Detector uses data from NASA's Kepler mission to build a binary classification model.

The goal is to distinguish between confirmed exoplanets and false-positive candidates based on recorded astronomical features.

This project explores the complete machine learning workflow, including data exploration, preprocessing, model training, evaluation, cross-validation, hyperparameter tuning, and feature importance analysis.

**Note:** This project classifies existing Kepler Objects of Interest using catalogue data. It does not directly detect new exoplanets from raw light curves.

## Dataset

This project uses the [NASA Kepler Exoplanet Search Results dataset](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results).

- Original source: NASA Kepler Mission
- Dataset provider: Kaggle
- Original dataset: 9,564 rows and 50 columns
- Licence: CC0 1.0 Universal (Public Domain Dedication)

### Target Classes

The original dataset contains three disposition categories. For this project, it was converted into a binary classification problem:

| Class | Label |
|---|---|
| Confirmed exoplanet | 1 |
| False positive | 0 |

Rows labelled as `CANDIDATE` were excluded.

The resulting dataset contains 7,316 observations:
- 2,293 confirmed exoplanets
- 5,023 false positives

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Machine Learning Workflow

### 1. Data Exploration
- Examined dataset structure, columns and data types.
- Investigated missing and duplicate values.
- Analysed target class distribution.
- Explored feature statistics and correlations.

### 2. Data Preprocessing
- Removed candidate observations.
- Converted the target into binary labels.
- Removed potential data leakage features and identifier columns.
- Removed completely empty features.
- Split the data into training and testing sets using stratified sampling (80/20).
- Handled missing values using median imputation.
- Standardised numerical features using StandardScaler.

### 3. Model Training
- Implemented Logistic Regression using Scikit-learn.
- Used DummyClassifier as a baseline comparison.
- Applied a Pipeline to combine imputation, scaling and model training.
- Used Stratified 5-Fold Cross-Validation.

### 4. Hyperparameter Tuning
Used GridSearchCV to tune the Logistic Regression regularisation parameter `C`.

The search evaluated the following values:

`0.001, 0.01, 0.1, 1, 10, 100`

F1-score was selected as the optimisation metric to balance precision and recall.

**Best hyperparameter:** `C = 100`

**Best cross-validation F1-score:** 87.60%

### 5. Model Evaluation and Interpretation
Evaluated the final tuned model using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Also investigated:
- Logistic Regression coefficients
- Permutation Feature Importance
- Feature Correlations and Multicollinearity

## Model Results

### Baseline vs Tuned Logistic Regression

| Metric | Baseline | Tuned Model |
|---|---:|---:|
| Accuracy | 90.71% | 91.33% |
| Precision | 86.46% | 87.39% |
| Recall | 83.44% | 84.53% |
| F1-score | 84.92% | 85.94% |

The tuned model improved all four evaluation metrics compared with the baseline model.

### Tuned Model Confusion Matrix

| | Predicted False Positive | Predicted Confirmed |
|---|---:|---:|
| Actual False Positive | 949 | 56 |
| Actual Confirmed | 71 | 388 |

The model correctly classified 949 false positives and 388 confirmed exoplanets in the test set.

## Feature Importance

Permutation Feature Importance was used to investigate how individual features contribute to the model's predictive performance.

The top features included:

| Feature | Mean Importance |
|---|---:|
| koi_teq | 0.113525 |
| koi_depth | 0.103051 |
| koi_impact | 0.054463 |
| koi_steff_err1 | 0.050000 |
| koi_srad_err1 | 0.037432 |

These values represent the average decrease in F1-score after randomly shuffling each feature. They are measures of predictive importance, not evidence of causation.

## Project Structure

```text
Exoplanet-Detector/
│
├── data/
│   └── cumulative.csv
│
├── src/
│   ├── data_exploration.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
```

### 2. Navigate to the project directory

```bash
cd Exoplanet-Detector
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Explore the dataset

```bash
python src/data_exploration.py
```

### 5. Train and evaluate the model

```bash
python src/evaluation.py
```

The evaluation script imports the training workflow, which performs cross-validation and hyperparameter tuning before evaluating the resulting model.

## Limitations

- The model uses existing Kepler catalogue features rather than raw light-curve measurements.
- It only distinguishes between confirmed exoplanets and false positives.
- Its performance depends on the quality and distribution of the training dataset.
- High evaluation scores do not guarantee the same performance on data from other missions or instruments.

## Future Improvements

- Experiment with additional classification algorithms.
- Explore more advanced feature selection techniques.
- Investigate raw Kepler light-curve data.
- Explore deep learning approaches such as 1D Convolutional Neural Networks (CNNs).

## Dataset Licence and Attribution

The dataset is dedicated to the public domain under the [CC0 1.0 Universal licence](https://creativecommons.org/publicdomain/zero/1.0/).

It may be copied, modified, distributed and used commercially without requesting permission.

Credit is given to NASA and the original dataset provider for making the data available.

---

**Disclaimer:** This is an educational machine learning project and is not an official NASA exoplanet detection system.