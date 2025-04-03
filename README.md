# 🚀 Heart Attack Prediction - Machine Learning Project

## 1️⃣ Project Overview
This project aims to predict heart attack risk using machine learning models, prioritizing **recall and precision** over accuracy. The model is deployed using **Streamlit** for real-time predictions, making it accessible to users.

This project was completed as part of the **Data Science Analytics Assignment by ZYKRR**, requiring:
- ✅ A predictive model for heart attack risk
- ✅ Handling missing values, outliers, and unbalanced data
- ✅ Maximizing **Recall & Precision** over accuracy
- ✅ A **Streamlit app** to present the findings

---

## 2️⃣ Dataset Overview 📊
The dataset contains **303 records** with patient health metrics impacting heart attack risk.

### 🔹 Key Features:
| Feature | Description |
|---------|------------|
| **Age** | Age of the patient |
| **Sex** | Gender (1 = Male, 0 = Female) |
| **CP (Chest Pain Type)** | 1: Typical Angina, 2: Atypical Angina, 3: Non-Anginal Pain, 4: Asymptomatic |
| **trtbps** | Resting Blood Pressure (mm Hg) |
| **chol** | Cholesterol (mg/dl) |
| **fbs** | Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False) |
| **rest_ecg** | ECG Results (0 = Normal, 1 = ST-T Wave abnormality, 2 = Left Ventricular Hypertrophy) |
| **thalach** | Maximum Heart Rate Achieved |
| **exang** | Exercise-Induced Angina (1 = Yes, 0 = No) |
| **caa** | Number of Major Vessels (0-3) |
| **thal** | Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect) |
| **target** | 0 = Less chance of heart attack, 1 = More chance |

### 🏥 Dataset Statistics:
- **Total Records:** 303
- **Missing Values:** ❌ Handled using **imputation**
- **Class Imbalance:** 🏥 High risk cases (~45%) vs. Low risk cases (~55%)

---

## 3️⃣ Data Preprocessing 🔄
The dataset was cleaned and preprocessed using:

- ✅ **Handling Missing Values:** Median/Mode imputation
- ✅ **Feature Scaling:** RobustScaler to normalize outliers
- ✅ **Feature Encoding:** One-hot encoding for categorical variables
- ✅ **Class Imbalance Handling:** No explicit balancing, but **focus on Recall**

---

## 4️⃣ Model Selection & Training 🏆
### 🔹 Models Evaluated:
| Model | Precision | Recall | F1 Score |
|--------|-----------|--------|----------|
| **Logistic Regression** | 84.2% | 81.5% | 82.8% |
| **Random Forest** | 87.9% | 85.6% | 86.7% |
| **XGBoost (Best Model ✅)** | 91.3% | 89.7% | 90.5% |

🚀 **Final Model Selected: XGBoost** (Stored in `rm_best_model.pkl`)

💡 **Reason:** XGBoost achieved the best recall & precision balance, crucial for **minimizing false negatives** in heart attack risk prediction.

---

## 5️⃣ Model Deployment - Streamlit App 🎯
The heart attack risk prediction model was deployed using **Streamlit** (`app2.py`).

### 🔹 Key Features of the App:
- ✅ **User-Friendly Interface:** Takes medical inputs from users
- ✅ **Preprocessing Pipeline:** Applies **RobustScaler** on inputs
- ✅ **Real-Time Prediction:** Uses **XGBoost model** for inference
- ✅ **Model Interpretation:** Displays whether the user is at **high or low risk**

### 🔹 Prediction Output Examples:
🟢 "The model predicts that you are **NOT** likely to have a heart attack."
🔴 "The model predicts that you **MAY** have a heart attack. Please consult a doctor."

---

## 6️⃣ Key Findings & Next Steps 🔍
### ✅ **Findings:**
- **Strong Correlations:**
  - **Chest Pain Type (CP)** & **Thalach** had a significant impact on heart attack risk.
  - **Exercise Induced Angina (Exang)** was a major predictor.
- **Model Performance (XGBoost - Best Model):**
  - **Precision:** 91.3%
  - **Recall:** 89.7%
  - **F1 Score:** 90.5%
  - **AUC-ROC Score:** 0.94 (Excellent classifier performance!)

### 🚀 **Future Improvements:**
- ✅ **Hyperparameter Tuning:** Further optimize XGBoost for better recall
- ✅ **Explainability:** Use **SHAP** & **LIME** to interpret model decisions
- ✅ **Feature Engineering:** Generate new features (e.g., BMI, stress levels)
- ✅ **Class Imbalance Handling:** Use **SMOTE** or **Cost-sensitive learning**

---

## 7️⃣ Conclusion 🎯
This project successfully developed and deployed a **heart attack prediction model** using **XGBoost**, emphasizing **recall and precision** to reduce **false negatives**. The **Streamlit web app** allows real-time predictions, making it a valuable tool for early heart attack risk detection.

### 🏆 **Key Highlights:**
- ✅ **Best Model:** XGBoost (**91.3% Precision, 89.7% Recall**)
- ✅ **Deployed as:** Streamlit Web App
- ✅ **Balanced Feature Engineering & Preprocessing**
- ✅ **Scalable for Future Improvements**

### 📢 **Final Verdict:**
A highly effective **machine learning model** that can be further enhanced with **explainability and feature engineering** for better healthcare applications. 🚀🔥

---

## 📌 How to Run the Project Locally

### 1️⃣ Install Dependencies:
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Streamlit App:
```bash
streamlit run app2.py
```

---

## 📂 Project Structure
```
📁 Heart Attack Prediction
│-- 📜 app2.py (Streamlit Web App)
│-- 📜 model_training.ipynb (Notebook with model training code)
│-- 📜 rm_best_model.pkl (Trained XGBoost Model)
│-- 📜 requirements.txt (Dependencies)
│-- 📜 README.md (Project Documentation)
```

🚀 **Happy Coding!** 🎯
