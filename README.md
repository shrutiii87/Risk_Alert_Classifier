# 🎯 Risk Alert Classifier

<img width="1200" height="500" alt="665135583-fbdfeeb8-a169-4464-bcbc-cf0bc29eb3f3" src="https://github.com/user-attachments/assets/9ff5b073-b88a-4a16-a021-088400edef9e" />


---

this project is to evaluate understanding of advanced supervised learning **classification** techniques, with a strong focus on imbalanced classification, resampling strategies (Under-Sampling, Over-Sampling, SMOTE, ADASYN), model generalization, hyperparameter tuning, and tree-based classification algorithms. Students will learn how to control overfitting, handle minority classes, and compare linear vs non-linear classifiers using real-world customer credit-risk data.

---

## 📂 Project Workflow

<img width="1200" height="900" alt="665142527-c85b3721-7b02-4c4c-ae50-d0fab08ddabe" src="https://github.com/user-attachments/assets/e2314ef2-7eb7-4454-b403-f913cb2ab250" />

---

## 📄 Problem Statement

working on a bank's credit-risk analytics team. The company holds a **customer risk dataset** of 4,600 customers and wants a **risk alert classifier** that can predict whether a customer is **risky (1)** or **safe (0)** from their demographic, credit-behaviour and transaction attributes.

task to build a baseline classifier, evaluate it with confusion-matrix metrics, handle the class imbalance problem with resampling techniques, compare tree-based models, tune the best model with Randomized and Grid Search, and deliver a final recommendation that **minimizes false negatives** — because missing a risky customer is the costliest mistake for the business.

---

## 🌐 View on streamlit :- 

<img width="800" height="450" alt="ezgif-8c90b305eb5586e0" src="https://github.com/user-attachments/assets/1fb835e3-bb5c-4ffa-9b35-a4602e389965" />



---


# 📂 Project Files

| 📄 File / Folder | 📌 Description |
|------------------|----------------|
| 📓 `Risk_Alert_Classifier.ipynb` | Main notebook — the complete, annotated classification, imbalanced-learning, tree, tuning and ROC-analysis pipeline |
| 📊 `Risk_Alert_Classifier_Dataset_4600 - Risk_Alert_Classifier_Dataset_4600.csv.csv` | Raw customer risk dataset (4,600 records) |
| 📘 `README.md` | Project documentation and workflow guide |
| 📂 `Visuals` | Folder of the output visuals/graphs |
| 📄 `Part A :- Conceptual_Foundation.pdf` | Project documentation of part :- A (Theory) 
| 📋 `Part H Final_Analysis_&_Reporting (Report).pdf` | Report documentation of part :- H (Report) |
| 🖥️ `app.py` | Main Streamlit application for the live deployment |
| ⚙️ `model.py.py` | Model file containing the trained prediction logic |
| 📦 `requirements.txt` | Required Python libraries and dependencies for the project |



---

## 🛠️ Tools Used

<div>

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Seaborn-Visuals-3776AB?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Imputation-KNN%20Imputer-EC4899?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Baseline-Logistic%20Regression-059669?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Resampling-Under%20%7C%20Over%20%7C%20SMOTE%20%7C%20ADASYN-16A34A?style=for-the-badge"/>

</div>

---

## 🎬 Project Demo

[![Watch Demo](https://img.shields.io/badge/Watch%20Demo-click%20to%20view-blue?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/YOUR-DEMO-VIDEO-LINK/view?usp=sharing)

📹 Click on the badge to watch the video . 

---

### 🧬 Dataset Structure — Customer Risk Dataset

| Field Name | Data Type | Description | Notes |
|------------|-----------|-------------|-------|
| `customer_id` | Integer | Unique identifier for each customer | Identifier only — not a real feature |
| `age` | Float | Age of the customer in years | Has missing values — KNN imputed |
| `gender` | Object | Gender of the customer | Categorical — not used in the models |
| `region` | Object | Region of the customer | Categorical — not used in the models |
| `employment_type` | Object | Employment category (Salaried, Self-Employed, etc.) | Categorical — not used in the models |
| `annual_income_inr` | Float | Annual income (₹) | Model feature — has missing values |
| `credit_score` | Float | Customer's credit score | Model feature — has missing values |
| `credit_utilization_ratio` | Float | Share of available credit being used | Model feature — has missing values |
| `missed_payments_12m` | Integer | Missed payments in the last 12 months | Model feature |
| `avg_late_payment_days` | Float | Average days a payment is late | Model feature |
| `monthly_transaction_count` | Integer | Number of transactions per month | Model feature |
| `monthly_spend_inr` | Float | Monthly spending (₹) | Model feature — has missing values |
| `cash_advance_count_6m` | Integer | Cash advances taken in the last 6 months | Model feature |
| `complaints_last_6m` | Integer | Complaints filed in the last 6 months | Model feature |
| `failed_login_attempts_3m` | Integer | Failed login attempts in the last 3 months | Model feature |
| `account_tenure_months` | Integer | How long the account has been open | Model feature |
| `last_transaction_date` | Object (date) | Date of the last transaction | Text date — not used as a model feature |
| `debt_balance_inr` | Integer | Outstanding debt balance (₹) | Model feature |
| `risk_status` | Integer | 1 = Risky, 0 = Safe | 🎯 **Target variable** (imbalanced: 12.11% risky) |

---


## 🧠 Part B : Dataset Understanding & Preparation

<img width="1200" height="500" alt="665135607-4a25a3d2-1aba-4553-a5fa-4470d1654d11" src="https://github.com/user-attachments/assets/d2f80372-c734-4c69-911b-177705863893" />


---

### 📋 Dataset Overview

```python
df = pd.read_csv("Risk_Alert_Classifier_Dataset_4600 - Risk_Alert_Classifier_Dataset_4600.csv.csv")
df.head(); df.tail(); df.describe(); df.info(); df.shape
```

**Output:** 4,600 rows × 19 columns · missing values in 7 columns (all under 5%)

💡 **Insight:** `credit_score` (216), `annual_income_inr` (166), `credit_utilization_ratio` (147), `employment_type` (144), `age` (140), `monthly_spend_inr` (129) and `region` (102) all have missing values, but none exceeds 5% of the rows. `customer_id` is only an identifier and `last_transaction_date` is text, so neither can be used directly as a numeric feature. 📊

---

### 7️⃣ Identify features and target variable

```python
X = df.drop("risk_status", axis=1)
y = df["risk_status"]
```

**Output:** 18 input features · target = `risk_status`

💡 **Insight:** The target is `risk_status` (1 = risky, 0 = safe) and every other column is a candidate feature. The target is binary, so this is a **binary classification** problem. `customer_id` sits in the feature list but carries no risk information, so it should be dropped before modelling. 🎯

---

### 8️⃣ Perform a train-test split while maintaining class distribution

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Output:** X_train (3680, 18) · X_test (920, 18)

💡 **Insight:** The 80/20 split gives 3,680 training and 920 testing rows, and `stratify=y` kept the class ratio almost identical — risky is **12.12%** in training and **12.07%** in testing. The test set holds about 111 risky customers, so minority-class metrics rest on a small number of cases. 🔀

---

### 9️⃣ Identify missing values and apply KNN Imputer for multivariate imputation

```python
imputer = KNNImputer(n_neighbors=5)

X_train[numeric_columns] = imputer.fit_transform(X_train[numeric_columns])
X_test[numeric_columns]  = imputer.transform(X_test[numeric_columns])
```

**Output:** 0 missing values in all numeric columns after imputation

💡 **Insight:** The imputer is **fit on the training set only** and then applied to the test set, so no test information leaks into training. It fills each missing value from the 5 most similar customers instead of a simple average, which preserves relationships between features. `KNNImputer` only works on numeric columns, so the missing values in `employment_type` and `region` are **not** filled by this step. ⚖️

---


## 📊 Part C : Baseline Classification Model

<img width="1200" height="500" alt="665135641-21418fab-697d-4205-bee0-dc7ac83bfe33" src="https://github.com/user-attachments/assets/7fe7500a-8943-437e-9494-0397ce82d140" />


---

### 🔟 Implement Logistic Regression as a baseline model

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train[numeric_columns])
X_test_scaled  = scaler.transform(X_test[numeric_columns])

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)
```

**Output:** Baseline Logistic Regression trained on 14 numeric features

💡 **Insight:** The scaler is **fit on the training set only**, which prevents data leakage. Scaling matters because Logistic Regression is sensitive to feature scale. Only the numeric columns are used, so `gender`, `region`, `employment_type` and `last_transaction_date` are not part of the model. 📐

---

### 1️⃣1️⃣ Generate and interpret the Confusion Matrix, Accuracy, Precision, Recall and F1-Score

```python
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:",  accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:",    recall_score(y_test, y_pred))
print("F1-Score:",  f1_score(y_test, y_pred))
```

**Output:** Confusion Matrix `[[809, 0], [0, 111]]` · Accuracy 1.0 · Precision 1.0 · Recall 1.0 · F1-Score 1.0

💡 **Insight:** All 809 safe and all 111 risky customers were classified correctly, so every metric is **1.0**. A perfect score on 920 rows is unusual — it suggests the numeric features separate the two classes almost linearly, but it also means this single split cannot show how the model behaves on harder cases. 🎯

---

### 1️⃣2️⃣ Identify Type-I and Type-II errors from the confusion matrix

```python
TN, FP, FN, TP = cm.ravel()

print("Type-I Error (False Positive):",  FP)
print("Type-II Error (False Negative):", FN)
```

**Output:** Type-I Error = 0 · Type-II Error = 0

💡 **Insight:** A **Type-I error (false positive) = 0** means no safe customer was wrongly flagged as risky, so no unnecessary alerts. A **Type-II error (false negative) = 0** means no risky customer was missed — in credit risk this is the costlier error, so this is the best possible result on this test set. ⚖️

---


## ⚖️ Part D : Handling Imbalanced Data

<img width="1200" height="500" alt="665135657-06a866a4-584a-49aa-b64e-9c5f249395d8" src="https://github.com/user-attachments/assets/884c5e48-8647-447c-9c4d-fdbcd65be86f" />


---

### 1️⃣3️⃣ Demonstrate the impact of class imbalance on model performance

```python
print(y.value_counts())
print(y.value_counts(normalize=True) * 100)
```

**Output:** Safe 4,043 (87.89%) · Risky 557 (12.11%)

💡 **Insight:** The class distribution is about **7 safe customers for every risky one**. A model that always predicted "safe" would reach roughly 88% accuracy while catching no risky customers, so accuracy alone is not a reliable measure here. In this notebook the imbalance did **not** hurt the baseline — recall and F1 are both 1.0. 📊

---

### 1️⃣4️⃣ Apply Under-Sampling, Over-Sampling, SMOTE and ADASYN, then retrain the model

```python
under  = RandomUnderSampler(random_state=42)
over   = RandomOverSampler(random_state=42)
smote  = SMOTE(random_state=42)
adasyn = ADASYN(random_state=42)
```

**Output:** Four rebalanced training sets, each used to retrain a Logistic Regression

💡 **Insight:** All four techniques are applied **only to the scaled training data** — the test set is untouched, so the comparison is fair. Under-sampling shrinks the safe class down to the risky class size and throws away safe-customer data; over-sampling duplicates risky rows; SMOTE and ADASYN create new synthetic risky rows instead. 🔁

---

### 1️⃣5️⃣ Compare performance before and after balancing (Recall, AUC-ROC, F1-Score)

| Model | Recall | F1-Score | AUC-ROC |
|-------|--------|----------|---------|
| Before Balancing | 1.0 | 1.0000 | 1.000000 |
| Under-Sampling | 1.0 | 0.9823 | 1.000000 |
| Over-Sampling | 1.0 | 0.9867 | 1.000000 |
| SMOTE | 1.0 | 0.9911 | 1.000000 |
| ADASYN | 1.0 | 0.9694 | 0.999978 |

💡 **Insight:** **Recall is 1.0 for every method**, and AUC-ROC is 1.0 for all except ADASYN. Since recall never dropped, the fall in F1 comes from lower precision — roughly 4 false positives for Under-Sampling, 3 for Over-Sampling, 2 for SMOTE and 7 for ADASYN. **SMOTE** was the best of the four and **ADASYN** the worst. Balancing did not improve anything because the baseline was already perfect; it only added a few false alarms. ⚖️

---


## 🌲 Part E : Tree-Based Classification Models

<img width="1200" height="500" alt="665135685-d4cef2c5-ebc8-42c8-8282-6b131a70e277" src="https://github.com/user-attachments/assets/b16f021f-3ca2-4179-ab88-b9b4293ced1e" />


---

### 1️⃣6️⃣ Implement Decision Tree Classifier

```python
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)
```

**Output:** Training Accuracy 1.0 · Testing Accuracy 0.9707

💡 **Insight:** The unrestricted tree (no `max_depth` set) reaches **100% training accuracy** — it has memorized the training data. Its 97.07% test accuracy implies about 27 wrong predictions out of 920, whereas Logistic Regression made 0. 🌱

---

### 1️⃣7️⃣ Analyze overfitting by comparing training and testing performance

```python
print("Training Accuracy:", accuracy_score(y_train, dt_train_pred))
print("Testing Accuracy:",  accuracy_score(y_test,  dt_test_pred))
```

**Output:** Training 1.0 · Testing 0.9707 — a gap of about **2.9 percentage points**

💡 **Insight:** The tree fits the training data perfectly but does not carry that performance over to unseen customers. This train–test gap is the classic signature of **overfitting**. ✂️

---

### 1️⃣8️⃣ Implement Random Forest Classifier

```python
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train_scaled, y_train)
```

**Output:** Training Accuracy 1.0 · Testing Accuracy 0.9967

💡 **Insight:** The Random Forest also reaches 100% training accuracy, but its test accuracy is **99.67%** — only about 3 mistakes in 920 rows. The train–test gap is just **0.33 points**, compared with about 2.9 for the single tree. 🌲

---

### 1️⃣9️⃣ Compare Decision Tree vs Random Forest in terms of accuracy and generalization

| Model | Training Accuracy | Testing Accuracy |
|-------|-------------------|------------------|
| Decision Tree | 1.0 | 0.9707 |
| Random Forest | 1.0 | 0.9967 |

💡 **Insight:** Random Forest cuts the error rate from about **2.9% to 0.33%**. Averaging 100 trees trained on different samples of the data reduces the variance that makes a single tree overfit, so the ensemble generalizes clearly better. 🥇

---


## 🔧 Part F : Hyperparameter Tuning

<img width="1200" height="500" alt="665135721-47ce1594-7516-412e-8e6e-b723e2194678" src="https://github.com/user-attachments/assets/96704769-8ba7-4cb0-b6c6-703f0f85e642" />


---

### 2️⃣0️⃣ Apply Randomized Search CV to optimize Decision Tree and Random Forest hyperparameters

```python
dt_params = {"max_depth": [3, 5, 7, 10, None],
             "min_samples_split": [2, 5, 10],
             "min_samples_leaf": [1, 2, 4]}

rf_params = {"n_estimators": [50, 100, 150],
             "max_depth": [5, 10, 15, None],
             "min_samples_split": [2, 5, 10],
             "min_samples_leaf": [1, 2, 4]}
```

**Output:** Best Decision Tree — `max_depth=None, min_samples_split=2, min_samples_leaf=2` · Best Random Forest — `n_estimators=150, max_depth=15, min_samples_split=10, min_samples_leaf=1`

💡 **Insight:** Both searches used `scoring="f1"` with 5-fold cross-validation, which suits an imbalanced target better than accuracy. Only 10 random combinations were tried (`n_iter=10`), so these are good settings but not necessarily the best possible. 🎛️

---

### 2️⃣1️⃣ Apply Grid Search CV for fine-tuning the best performing model

```python
grid_params = {"max_depth": [5, 10, 15],
               "min_samples_split": [2, 5],
               "min_samples_leaf": [1, 2]}

grid = GridSearchCV(RandomForestClassifier(n_estimators=100, random_state=42),
                    param_grid=grid_params, cv=5, scoring="f1")
```

**Output:** Best Parameters: `max_depth=15, min_samples_split=5, min_samples_leaf=1`

💡 **Insight:** Grid Search agrees with the randomized search on `max_depth=15` and `min_samples_leaf=1`, so those two settings look stable. The grid fixed `n_estimators=100`, so it fine-tunes tree depth and split rules only. 🎯

---

### 2️⃣2️⃣ Compare tuned vs untuned model performance

| Model | Test Accuracy | F1-Score |
|-------|---------------|----------|
| Untuned Random Forest | 0.9967 | 0.9865 |
| Tuned Random Forest | 0.9978 | 0.9910 |

💡 **Insight:** On 920 test rows the improvement is about **3 errors down to 2**, so tuning fixed one prediction. The gain is real but very small, because the untuned Random Forest was already close to perfect. ⚖️

---


## 📈 Part G : Model Evaluation & ROC Analysis

<img width="1200" height="500" alt="665135790-9ef72594-ceda-4da8-935b-da674c537a0a" src="https://github.com/user-attachments/assets/e32e18af-56d8-4b59-ae6d-d4b9fe7678a4" />


---

### 2️⃣3️⃣ Plot and interpret the ROC Curve for all models

```python
for name, m in models.items():
    y_prob = m.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=name)
```

**Output:** ROC curves for Logistic Regression, Decision Tree, Random Forest and Tuned Random Forest

💡 **Insight:** Logistic Regression, Random Forest and Tuned Random Forest hug the top-left corner — the ideal ROC shape. The **Decision Tree** curve is clearly lower, with a single corner at about 86% true positive rate, typical of a tree that gives only hard 0/1 predictions. All curves sit far above the diagonal line that represents random guessing. 📉

---

### 2️⃣4️⃣ Compute and compare AUC-ROC scores

| Model | AUC-ROC |
|-------|---------|
| Logistic Regression | 1.0000 |
| Random Forest | 0.99989 |
| Tuned Random Forest | 0.99989 |
| Decision Tree | 0.9250 |

💡 **Insight:** The Decision Tree has the lowest AUC by a clear margin, so it ranks customers by risk worst. Tuning left the Random Forest AUC unchanged — its small improvement shows up in the final predictions, not in how customers are ranked. 🏆

---

### 2️⃣5️⃣ Select the best final model based on business requirements (minimizing false negatives)

| Model | Recall | Risky customers missed (of 111) |
|-------|--------|--------------------------------|
| **Logistic Regression** | **1.0000** | **0** |
| Random Forest | 0.9910 | 1 |
| Tuned Random Forest | 0.9910 | 1 |
| Decision Tree | 0.8649 | 15 |

💡 **Insight:** The code picks the model with the highest recall, which is **Logistic Regression** — matching the goal of minimizing false negatives. The Decision Tree is the worst choice: it would let about 15 risky customers through. Caveat: the gap between 0 and 1 missed customers rests on only 111 risky cases in one test split, so cross-validation should confirm this before deployment. 🚀

---


## 📂 Project Workflow

1. **Dataset Understanding** → Overview, missing values, features and target
2. **Train/Test Split** → 80/20 stratified split (3,680 / 920 rows)
3. **Preprocessing** → KNN Imputer (fit on training data only) + StandardScaler
4. **Baseline Model** → Logistic Regression with confusion-matrix metrics
5. **Class Imbalance** → Under-Sampling, Over-Sampling, SMOTE and ADASYN compared
6. **Tree-Based Models** → Decision Tree (overfitting analysis) and Random Forest
7. **Hyperparameter Tuning** → RandomizedSearchCV then GridSearchCV
8. **Model Evaluation** → ROC curves, AUC-ROC and recall-based model selection
9. **Final Recommendation** → Best model chosen to minimize false negatives

---

## 📈 Results & Insights

- ✅ **Four model variants** built and compared — Logistic Regression, Decision Tree, Random Forest, Tuned Random Forest
- ✅ **Logistic Regression** selected as the final model — Accuracy 1.0, Recall 1.0, AUC-ROC 1.0, **zero false negatives**
- ✅ **Random Forest** is a strong backup — 99.67% accuracy, AUC 0.99989, only 1 risky customer missed
- ✅ **Class imbalance (12.11% risky)** did not hurt the baseline — all four resampling techniques kept recall at 1.0 but slightly lowered F1 (SMOTE best, ADASYN worst)
- ✅ **Decision Tree overfits** — 1.0 train vs 0.9707 test; Random Forest cut the error rate from 2.9% to 0.33%
- ✅ **Tuning** lifted Random Forest accuracy from 0.9967 to 0.9978 and F1 from 0.9865 to 0.9910
- ✅ **Decision Tree is the worst choice for this business** — it would miss about 15 of the 111 risky customers

---

## 📌 Expected Outcomes

- Understand how to **plan and execute a complete supervised-learning classification workflow**
- Apply **KNN multivariate imputation** and leakage-free scaling on imbalanced customer data
- Evaluate classifiers with **confusion matrix, accuracy, precision, recall, F1 and error types (Type-I / Type-II)**
- Handle **class imbalance** with Under-Sampling, Over-Sampling, SMOTE and ADASYN and judge when they help
- Build **tree-based models**, diagnose overfitting, and tune them with **RandomizedSearchCV and GridSearchCV**
- Interpret **ROC curves and AUC-ROC** and select a final model against a **business requirement (minimizing false negatives)**

---

## ⚙️ Installation & Setup

```bash
# clone the repository
git clone https://github.com/yourusername/risk-alert-classifier.git
cd risk-alert-classifier

# create an isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn jupyter

# launch the notebook
jupyter notebook Risk_Alert_Classifier.ipynb
```

---

## 🚀 Future Scope

- [ ] Remove `customer_id` from the feature set before deployment
- [ ] Encode and test the unused categorical columns (`gender`, `region`, `employment_type`) and features derived from `last_transaction_date`
- [ ] Confirm the perfect baseline score with cross-validation across multiple splits
- [ ] Prune the Decision Tree (`max_depth`, `min_samples_leaf`) to reduce its overfitting
- [ ] Add Gradient Boosting and XGBoost as further baselines
- [ ] Tune the decision threshold to trade precision for even lower false-negative risk
- [ ] Deploy the best model behind a simple risk-alert API

---

## 🙏 Thank You

Thank you for taking the time to explore this project!
Your feedback, suggestions, and contributions are always welcome.

⭐ If you found this project helpful, don't forget to **star the repository** and share it with others.
