# Date a Scientist: Predicting Body Type from Online Dating Profiles

This project explores the use of data science techniques to predict a person's body type based on features extracted from their online dating profile. It began as a multi-class classification task, transformed to binary classification after some cardinality reduction, involving structured data and categorical variables, with a strong emphasis on data preprocessing and dimensionality reduction.

---

## Project Structure

- `01_eda.ipynb`: Initial exploratory data analysis.
- `02.1_preprocessing1.ipynb`: Preprocessing Part 1 - Handling missing values, variable imputation and first cardinality reduction.
- `02.2_preprocessing2.ipynb`: Preprocessing Part 2 - Preprocessing for classification problem, imputation, scoping the problem, second cardinality reduction, encoding and transforming variables, creating predictors and labels subsets (X and y). Creation of new variables as a combination of others.
- `03_pca.ipynb`: Principal Component Analysis for dimensionality reduction.
- `04.1_xgboost_no_pca.ipynb`: XGBoost model trained on original preprocessed features.
- `04.2_xgboost_pca.ipynb`: XGBoost model trained on PCA-reduced features.

---

## Objective

To build a machine learning model capable of classifying users into one of four body type categories based on their profile information, including lifestyle habits, diet, and other attributes.

---

## Techniques Used

- **Data Cleaning & Preprocessing**
  - Handling missing values with a defined function
  - Categorical encoding (one-hot, label, binary)
  - Feature selection
  - Scaling/transforming numerical features
- **Dimensionality Reduction**
  - PCA (Principal Component Analysis) with 95% explained variance
- **Modeling**
  - XGBoost classifier
  - Class balancing using RandomOverSampler, SMOTE and manually assigning weights.
  - Hyperparameter tuning with GridSearchCV
- **Evaluation**
  - Accuracy, precision, recall, f1-score
  - Confusion matrix analysis

---

## Dataset

The dataset comes from Codecademy as a Final Project for the Data Science and Machine Learning career path. For this project, I focused on a cleaned and filtered version of the dataset with:
- ~50,000 profiles
- 18 features, turned into 40 after transformation and encoding, then regrouped as 11 principal components
- 2 body-type classes

---

### Threshold Optimization
- Adjusted decision threshold to maximize recall and F1.
- Plotted:
  - ROC Curve
  - AUC
  - Threshold vs Precision / Recall / F1

### Final Observations
- Removing dominant features like `age` and `height` did not significantly improve predictions.
- Recall improved for minority class at the expense of precision.
- Overall, models achieved reasonable separation, though precision for minority class remains low.

## 📊 Final Model Performance (Test Set)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
|   0   |   0.91    |  0.59  |   0.72   |  7611   |
|   1   |   0.23    |  0.69  |   0.35   |  1381   |

- **Accuracy**: 0.61  
- **Macro avg**: Precision: 0.57, Recall: 0.64, F1: 0.53  
- **Confusion Matrix**:

## ⚙️ Requirements

- Python 3.9+
- scikit-learn
- xgboost
- pandas, numpy
- seaborn, matplotlib
