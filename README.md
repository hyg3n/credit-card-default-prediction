# Credit Card Default Prediction – Model Comparison

This project implements an end-to-end **credit risk classification workflow** on a real-world credit card default dataset.  
The goal is to predict whether a client will default on a credit card payment (`Y = 1`) or not (`Y = 0`), using 23 demographic and financial features (`X1`–`X23`).

The notebook walks through data exploration, model building, hyperparameter tuning, threshold optimisation, and feature interpretation, with a focus on handling **class imbalance** in a realistic risk-modelling setting.

---

## Dataset

The data consists of two CSV files with a predefined split:

- `creditdefault_train.csv` – training set (15,000 rows)  
- `creditdefault_test.csv` – test set (15,000 rows)

**Target variable**

- `Y`: credit card default indicator  
  - `1` = client defaulted  
  - `0` = no default  

**Input features (X1–X23)** include:

- **Credit limit & payments**  
  - `X1`: credit amount (NT dollars)  
  - `X12–X17`: bill statement amounts (April–September 2005)  
  - `X18–X23`: previous payment amounts  

- **Customer profile**  
  - `X2`: gender (1 = male, 2 = female)  
  - `X3`: education level  
  - `X4`: marital status  
  - `X5`: age  

- **Payment history (categorical, ordinal)**  
  - `X6–X11`: repayment status in previous months  
    - original documentation: `-1` = paid duly, `1`–`9` = months of delay  
    - additional undocumented codes (`-2`, `0`) are analysed and kept as part of the modelling process.

The class distribution is **imbalanced**: roughly **22% defaulters** vs **78% non-defaulters**.

---

## Methods & Modelling Workflow

The notebook follows a structured workflow:

1. **Exploratory Data Analysis (EDA)**
   - Shape, summary statistics, and class distribution.
   - Histograms and basic visualisations.
   - Frequency tables for categorical features (X2, X3, X4, X6–X11).
   - Identification and discussion of **undocumented values** (e.g. `X3 = 0,5,6`; `X4 = 0`; `X6–X11 = -2, 0`).

2. **Handling Class Imbalance**
   - Recognised imbalance (~22% positive class).
   - **F1-score** chosen as the **primary metric** for:
     - Hyperparameter tuning (`scoring="f1"` in cross-validation).
     - Final model comparison on the test set.
   - Accuracy is still reported, but not used for model selection.

3. **Feature Processing**
   - Features/labels split: `X1–X23` as predictors, `Y` as target.
   - For **k-NN**:
     - One-hot encoding of `X2` (gender) and `X4` (marital status).
     - Standardisation of numeric/ordinal features using `StandardScaler`.
   - For **tree-based models** (Random Forest, AdaBoost):
     - Raw numeric and ordinal features are used directly (trees are scale-invariant).

4. **Models Implemented**
   - **k-Nearest Neighbours (k-NN)**
   - **Random Forest**
   - **AdaBoost** (with `DecisionTreeClassifier` as base estimator, SAMME algorithm)

5. **Hyperparameter Tuning (Cross-Validation)**
   - Performed on the **training set** using 5-fold cross-validation.
   - Main scoring metric: **F1-score**.
   - For each model, a numeric hyperparameter is varied and visualised:
     - k-NN: number of neighbours `k`.
     - Random Forest: number of trees `n_estimators`.
     - AdaBoost: number of estimators `n_estimators`.

6. **Model Evaluation**
   - Evaluated on the **held-out test set**:
     - **Accuracy**
     - **Precision**
     - **Recall**
     - **F1-score**
     - **ROC AUC**
   - Confusion matrices and ROC curves for each model.
   - Bar chart comparison across all models and metrics.

7. **Threshold Optimisation**
   - For each model, the decision threshold on predicted probabilities is varied from 0.10 to 0.90.
   - F1-score is tracked as a function of the threshold.
   - The notebook reports the **optimal threshold** and **best F1-score** per model and compares it with the default threshold of 0.5.

8. **Feature Importance & Undocumented Values**
   - Random Forest feature importances are plotted.
   - Payment history features (`X6–X11`) are found to be most influential, despite containing undocumented values (`-2`, `0`).
   - The impact and risks of relying on these ambiguous codes are discussed, including how tree-based models may exploit them as useful split points.

---

## Key Results (Test Set)

Using the default probability threshold of 0.5:

- **Random Forest**
  - F1-score: **0.5476**  
  - Recall: **0.5856**  
  - Precision: **0.5143**  
  - AUC: **0.7775**  
  - Best overall balance of recall, F1, and discriminative power.

- **AdaBoost**
  - F1-score: **0.4775**  
  - Recall: **0.3731**  
  - Precision: **0.6631**  
  - AUC: **0.7208**  
  - More conservative, with higher precision but lower recall.

- **k-NN (k = 7)**
  - F1-score: **0.4347**  
  - Recall: **0.3412**  
  - Precision: **0.5989**  
  - AUC: **0.7159**  
  - Highest accuracy but weaker performance on the minority (default) class.

After **threshold optimisation**, AdaBoost and k-NN achieve improved F1-scores at lower thresholds, highlighting their tendency to underpredict defaults at the standard 0.5 cutoff.

The **Random Forest** model is selected as the **best model**, based on its highest F1-score and strong AUC.

---

## How to Use This Notebook

Minimal setup is needed to run this project:

1. **Requirements**
   - Python 3.x
   - JupyterLab / Jupyter Notebook
   - Common Python libraries:
     - `pandas`
     - `numpy`
     - `matplotlib`
     - `seaborn`
     - `scikit-learn`

2. **Steps**
   - Place `creditdefault_train.csv` and `creditdefault_test.csv` in the same directory as the notebook.
   - Open the notebook (e.g. `credit_default_models.ipynb`) in JupyterLab.
   - Run all cells from top to bottom.

If you use a different environment (e.g. VS Code, local Jupyter, or a cloud notebook), make sure the required libraries are installed and the file paths to the CSVs are correct.

---

## Skills Demonstrated

- Supervised learning for **credit risk / default prediction**
- Handling **imbalanced datasets** and selecting appropriate metrics (F1, Recall, AUC)
- **Hyperparameter tuning** with `GridSearchCV` (k-NN, Random Forest, AdaBoost)
- **Threshold optimisation** for probability-based classifiers
- Feature preprocessing:
  - One-hot encoding of categorical variables
  - Feature scaling for distance-based models
- Model interpretation via **feature importance** and correlation analysis
- Critical analysis of **data quality issues** (undocumented categorical codes) and their impact on model behaviour
- Clear, structured **data mining workflow documentation** in Jupyter

---

## Project Structure

```text
.
├── creditdefault_train.csv
├── creditdefault_test.csv
├── Credit_Risk_3Classifier_Comparison.ipynb   # Main notebook (EDA, modelling, evaluation, interpretation)
├── README.md
└── LICENSE                       # CC BY 4.0 license
```
