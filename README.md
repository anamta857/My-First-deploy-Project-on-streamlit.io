
# 💼 Salary Prediction Using Traditional ML Techniques

## 🎯 Project Goal
The goal of this project is to build a machine learning regression model that can predict a person's salary based on their years of experience. This project is part of the AI Internship Program at DIGIPEX Solutions.

---

## 📝 Project Overview
This project applies traditional machine learning techniques to solve a real-world regression problem:
- Explore and preprocess the dataset
- Engineer useful features
- Train and evaluate regression models
- Deploy the final model using Streamlit
- Host the interactive web app on Render

---

## 📊 Dataset
**File:** `Salary_Data.csv`  
**Columns:**
- **YearsExperience** — Total years of work experience
- **Salary** — Corresponding salary in USD

---

## 📌 Project Steps

### **Step 1: Data Exploration & Preprocessing**
- Loaded dataset using Pandas
- Verified there were no missing values
- No categorical encoding needed (all numeric)
- Applied StandardScaler on YearsExperience  
✅ *Dataset is clean and ready for training.*

---

### **Step 2: Feature Engineering**
- Checked correlation between YearsExperience and Salary
- Scatter plot showed strong positive linear relationship
- Correlation coefficient = **0.978**
- Decided no derived features were necessary  
✅ *YearsExperience is the main predictive feature.*

<img width="446" height="372" alt="task4 1" src="https://github.com/user-attachments/assets/2cfe2976-a732-4642-9fb2-b94e01037025" />

<img width="567" height="468" alt="task 4 2" src="https://github.com/user-attachments/assets/f33eb154-7bce-4045-bd1a-d1ded4dfe889" />



---

### **Step 3: Model Building**
Trained 3 regression algorithms:
1. Linear Regression
2. Random Forest Regressor
3. XGBoost Regressor  
✅ *All models trained successfully.*

---

### **Step 4: Model Evaluation**
Metrics used: MAE, MSE, RMSE, R² Score

| Model              | MAE       | RMSE      | R² Score |
|--------------------|-----------|-----------|----------|
| Linear Regression  | 6286.45   | 7059.04   | **0.9024** |
| Random Forest      | 6651.76   | 7521.74   | 0.8892   |
| XGBoost            | 8912.31   | 10168.80  | 0.7975   |

📌 **Best Model:** Linear Regression (highest accuracy, lowest error).

---

### **Step 5: Streamlit App**
- Interactive UI for predicting salary
- User can enter **Years of Experience**, **Education Level**, and **Skill Level**
- Output: Predicted salary

---

### **Step 6: Deployment**
The app is hosted on Streamlit.io for public use.  

<img width="824" height="567" alt="task 4" src="https://github.com/user-attachments/assets/d6bca514-b378-4519-a5a6-1ad0ccbb0f2e" />

<img width="1080" height="652" alt="task 4 output" src="https://github.com/user-attachments/assets/1851cd58-2632-48f6-9846-32c9cc61819a" />




---

## 📦 Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
