# Credit Card Fraud Detection

Exploratory analysis and preprocessing of European cardholder transactions to prepare a fraud classification model.

## Problem

The dataset has 284,807 transactions, of which only **492 (0.17%) are fraudulent**. With such an extreme class imbalance, a model can reach 99.8% accuracy while never detecting a single fraud. So most of the work goes into understanding and rebalancing the data before training.

## What the notebook covers

1. **Data quality checks:** schema, summary statistics, null values.
2. **Class imbalance analysis:** class distribution and the distributions of `Amount` and `Time`.
3. **Scaling:** `RobustScaler` on `Amount` and `Time`, which is less sensitive to outliers than standard scaling.
4. **Stratified splitting:** `StratifiedKFold` keeps the fraud ratio the same in every fold.
5. **Random undersampling:** builds a balanced 50/50 subsample so correlations are not dominated by the majority class.
6. **Correlation analysis:** compares the correlation matrices of the imbalanced data and the balanced subsample, and looks at the features most negatively (V17, V14, V12, V10) and positively (V11, V4, V2, V19) correlated with fraud.
7. **Outlier removal:** an IQR-based filter on the most predictive features.

**Next steps:** train and compare classifiers (logistic regression, tree ensembles), evaluate them with precision-recall AUC instead of accuracy, and compare undersampling with SMOTE oversampling.

## Data

[Credit Card Fraud Detection, Kaggle (ULB Machine Learning Group)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Features `V1`–`V28` are PCA components; they are anonymized for confidentiality.

Download `creditcard.csv.zip` and place it in `data/`. The dataset is not versioned in this repository.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook fraud_detection.ipynb
```
