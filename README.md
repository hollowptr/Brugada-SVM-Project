# Brugada Syndrome Detection Using Support Vector Machines (SVM)

In this project, I developed a Support Vector Machines (SVM) classifier to detect **Brugada Syndrome** using ECG signals. The main objective of this project is to investigate how machine learning can support early detection of Brugada Syndrome, contributing to improved clinical decision-making and prevention of sudden cardiac death.

## Table of Contents

1.  Introduction
2.  Background (SVM)
3.  Problem Statement: Brugada Syndrome Detection
4.  Dataset Description (Brugada-HUCA ECG Dataset)
5.  Methodology
    *   Exploratory Data Analysis (EDA)
    *   Feature and Target Variable Definition
    *   Data Splitting (Train-Test Split) and Class Imbalance Handling (SMOTE)
    *   Feature Scaling
    *   SVM Model Training with Various Kernels
    *   Stratified Cross-Validation
    *   Hyperparameter Optimization with GridSearchCV
6.  Model Evaluation
    *   Confusion Matrix
    *   Classification Metrics (Precision, Recall, F1-Score)
    *   ROC-AUC Curve
7.  Results and Conclusion

## 1. Introduction

This project focuses on developing a machine learning model to detect Brugada Syndrome, a genetic heart condition that can lead to dangerous ventricular arrhythmias and sudden cardiac death. By utilizing electrocardiogram (ECG) signals and the SVM algorithm, we aim to create a supportive tool that can improve diagnostic accuracy and speed, especially when facing real-world data challenges such as class imbalance.

## 2. Background (Support Vector Machines)

**Support Vector Machines** (SVMs) are powerful machine learning algorithms for classification and regression tasks. In classification, SVM works by finding an optimal *hyperplane* that distinctly separates data points from different classes with the largest margin. This algorithm is known for its ability to handle complex data, including non-linearly separable data, through the use of the **_kernel trick_**.

In this project, various kernels such as linear, polynomial, RBF (Radial Basis Function), and sigmoid will be explored to find the best model for classifying ECG signals.

## 3. Problem Statement: Brugada Syndrome Detection

In this project, we aim to classify ECG signals into categories: `Brugada Syndrome`, `Normal`, and `Ambiguous`. Brugada Syndrome represents a minority positive class, while normal ECG signals form the majority negative class. This class imbalance makes the classification task more challenging and requires special handling.

To address this problem, the Support Vector Machines (SVM) classification algorithm will be implemented using Python and Scikit-Learn. Our goal is to build a model that can accurately distinguish between Brugada, Normal, and Ambiguous ECG patterns.

## 4. Dataset Description (Brugada-HUCA ECG Dataset)

The dataset used is the **Brugada-HUCA dataset**, publicly available on PhysioNet: [Brugada-HUCA Dataset (PhysioNet)](https://physionet.org/content/brugada-huca/1.0.0/).

This dataset contains 12-lead electrocardiogram (ECG) recordings specifically designed for the study and classification of Brugada Syndrome. There are a total of 363 subjects, consisting of:
*   69 patients diagnosed with Brugada Syndrome (Class 1)
*   287 healthy control subjects (Class 0)
*   7 ambiguous cases (Class 2)

Each ECG recording has a duration of approximately 12 seconds and is sampled at a frequency of 100 Hz. This dataset exhibits a significant **class imbalance** problem, which is a focus during the _preprocessing_ and model training process.

**Attribute Information:**
The dataset consists of two main components:
1.  **ECG waveform data** (`.dat` and `.hea` files)
2.  **Metadata (CSV file)** containing `patient_id`, `brugada` (target label: 0=Normal, 1=Brugada, 2=Ambiguous), and other clinical indicators.

## 5. Methodology

### Exploratory Data Analysis (EDA)

EDA is performed to understand the data structure, class distribution, and identify potential issues such as missing values or _outliers_. Visualizations like _countplot_ are used to show the dominant class imbalance in Class 0 (Normal).

### Feature and Target Variable Definition

The feature set (`X`) consists of `'basal_pattern'` and `'sudden_death'`, while the target variable (`y`) is `'brugada'`. The `'patient_id'` column is ignored as it only serves as an identifier.

### Data Splitting (Train-Test Split) and Class Imbalance Handling (SMOTE)

The data is split into training and testing sets with an 80:20 ratio using `stratify=y` to maintain class distribution. To address severe class imbalance, the **SMOTE** (Synthetic Minority Over-sampling Technique) oversampling technique is applied to the training data.

### Feature Scaling

Features are scaled using `StandardScaler` to have a mean of zero and a variance of one. This is an important step for most _machine learning_ algorithms, including SVMs, so that features with different value ranges do not dominate the learning process.

### SVM Model Training with Various Kernels

SVM models are trained using `OneVsRestClassifier` to handle multi-class classification with various kernel types:
*   **RBF (Gaussian) Kernel:** Explored with different `C` values (1.0, 100.0, 1000.0).
*   **Linear Kernel:** Explored with different `C` values (1.0, 100.0, 1000.0).
*   **Polynomial Kernel:** Explored with different `C` values (1.0, 100.0, 1000.0) and `degree=3`.
*   **Sigmoid Kernel:** Explored with different `C` values (1.0, 100.0, 1000.0).

Research shows that increasing `C` beyond a certain point does not always lead to a significant increase in model accuracy, especially for certain kernels.

### Stratified Cross-Validation

**Stratified k-fold Cross-Validation** with `shuffle=True` is used for a more robust model evaluation, especially due to class imbalance. This ensures that each fold has the same class proportion as the overall dataset. The average cross-validation score for the linear kernel is **0.7766** and for the RBF kernel is **0.7903**.

### Hyperparameter Optimization with GridSearchCV

**GridSearchCV** is used to find the optimal hyperparameter combination for various kernels. This process identified `{'C': 1, 'degree': 2, 'gamma': 0.01, 'kernel': 'poly'}` as the best parameters with the best accuracy score on the training set of **0.7897**, and on the test set of **0.7945**.

## 6. Model Evaluation

Model evaluation is performed comprehensively using various metrics, considering the imbalanced nature of the data.

### Confusion Matrix

The confusion matrix shows that the model tends to be heavily biased towards the majority class (Normal/Class 0), with 57 correct predictions for Class 0. However, the model struggles to identify Class 1 (Brugada) and Class 2 (Ambiguous), where almost all minority cases are predicted as Class 0. This highlights the limitations of accuracy as a single metric in cases of class imbalance.

### Classification Metrics (Precision, Recall, F1-Score)

*   **Without Class Balancing:** The model is only able to predict Class 0 well. Precision, Recall, and F1-Score for Classes 1 and 2 are close to 0.
*   **With `class_weight='balanced'`:** No significant improvement in minority class performance.
*   **With SMOTE:** The use of SMOTE slightly increased the *recall* for the Brugada Class (from 0.00 to 0.07) and precision to 0.33, showing a slight improvement in identifying the minority class, although the Ambiguous Class is still not well identified.

Example classification report after SMOTE:
```
               precision    recall  f1-score   support

   Normal (0)       0.81      0.98      0.89        58
  Brugada (1)       0.33      0.07      0.12        14
Ambiguous (2)       0.00      0.00      0.00         1

     accuracy                           0.79        73
    macro avg       0.38      0.35      0.34        73
 weighted avg       0.71      0.79      0.73        73
```

### ROC-AUC Curve

The calculated ROC AUC score is **0.4904**. This value indicates a very low discrimination ability of the model, especially in multi-class scenarios with imbalanced data, where the model may struggle to distinguish minority classes from majority classes. The ROC curve also confirms that the model's performance is only slightly better than random guessing.
