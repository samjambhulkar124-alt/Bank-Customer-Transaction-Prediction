# Bank-Customer-Transaction-Prediction
# Bank Customer Transaction Prediction

## 📌 Project Overview
The objective of this project is to predict which bank customers will make a specific transaction in the future, irrespective of the amount of money transacted. This is a binary classification problem where the target variable is highly imbalanced.

Predicting customer transactions allows banking institutions to optimize their marketing campaigns, improve customer retention strategies, and accurately allocate operational resources.

---

## 📂 Project Structure
- **Domain:** Banking
- **Dataset:** 200,000 customer records with 200 anonymized numeric features (`var_0` to `var_199`), an `ID_code`, and a binary `target` column.
- **Target Value Meanings:**
  - `0`: Customer will **not** make a transaction in the future.
  - `1`: Customer **will** make a transaction in the future.

---

## 📊 Task 1: Data Analysis Summary
A thorough exploratory and statistical analysis was conducted on the dataset, revealing the following key characteristics:
* **Data Quality:** Excellent. The dataset contains **0 missing values** and **0 duplicate rows**, meaning no raw cleaning or imputations were required.
* **Severe Class Imbalance:** Only **10.05%** of the customers belong to Class 1 (will transact), while **89.95%** belong to Class 0. Because of this, standard *Accuracy* cannot be used as a standalone metric to judge model performance.
* **Feature Relationships:** The 200 anonymized features are completely independent of each other, with a maximum mutual correlation of just **0.0098** (no multicollinearity).
* **Weak Predictive Signal:** No single feature correlates strongly with the target. The highest correlations belong to `var_81` (0.081) and `var_139` (0.074). This indicates that the predictive signal is weak and spread thinly across all variables, requiring models capable of combining complex, non-linear interactions.

---

## 🤖 Task 2 & 3: Predictive Modeling & Comparison
The data was split using a stratified approach into **80% Training (160,000 rows)** and **20% Validation (40,000 rows)** to preserve the 90/10 target distribution. Models were trained utilizing class-balancing techniques to prevent bias toward the majority class. 

### Model Performance Metrics

| Model | ROC-AUC | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| 🏆 **LightGBM** | **0.8814** | **0.3494** | **0.7512** | **0.4770** |
| 🥈 Logistic Regression | 0.8599 | 0.2865 | 0.7754 | 0.4184 |
| 🥉 Random Forest | 0.7810 | 0.3215 | 0.4271 | 0.3669 |

### Production Recommendation
**LightGBM** is recommended as the final production model. It achieves the highest **ROC-AUC (0.8814)** and **F1-Score (0.4770)**. In a banking context, missing a transaction opportunity (False Negative) is typically more costly than a false alarm. LightGBM balances this trade-off effectively by discovering **75.12% of actual transactors** while keeping the targeting precision 3.5x higher than a random guess.

---

## 🛠️ Task 4: Report on Challenges Faced

| Challenge | Impact on Project | Resolution Technique & Rationale |
| :--- | :--- | :--- |
| **Severe Class Imbalance** | Standard accuracy metrics become completely misleading. A dummy model predicting "always 0" would get 90% accuracy but 0% recall. | Utilized stratified train/test splitting. Adjusted algorithmic cost functions via `class_weight="balanced"` (Logistic Regression/Random Forest) and `is_unbalance=True` (LightGBM). Evaluated models using **ROC-AUC** as the core metric. |
| **Weak and Dispersed Signal** | Linear models struggle to identify standalone patterns because no single feature correlates highly with the target. | Leveraged **LightGBM (Gradient Boosting)**, which sequentially combines weak decision trees to isolate complex, non-linear boundaries distributed across all 200 features. |
| **Varying Feature Scales** | Distance-based and linear models (like Logistic Regression) can get distorted by mismatched standard deviations across features. | Applied `StandardScaler` to normalize the data. Crucially, the scaler was **fitted only on the training set** and transformed on the validation set to eliminate data leakage. |

---

## 🚀 How to Run the Project
1. **Clone the Repository:**
   ```bash
   git clone https://github.com
   cd YOUR_REPOSITORY_NAME
   ```
2. **Install Required Libraries:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn lightgbm
   ```
3. **Open the Notebook:**
   Run and view the implementation inside your Jupyter environment:
   ```bash
   jupyter notebook
   ```

