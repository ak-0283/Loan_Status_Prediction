# 💳 Loan Status Prediction using SVM

🏦 This project predicts whether a **loan will be approved (1)** or **not approved (0)** using **Machine Learning (Support Vector Machine)**.

---

## ⚙️ Workflow

1️⃣ **Loan Dataset** 📊 – Load dataset with 614 rows & 13 columns.
2️⃣ **Data Preprocessing** 🧹 – Encode categorical values & clean the dataset.
3️⃣ **Exploratory Data Analysis** 📈 – Visualize relations between features & loan status.
4️⃣ **Train-Test Split** ✂️ – Split into training and testing sets.
5️⃣ **Support Vector Machine (SVM) Model** 🧠 – Train the model to classify loan approvals.
6️⃣ **Prediction** 🔮 – Feed new data → predict loan status.

---

## 📦 Importing Dependencies

```python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
```

---

## 📊 Dataset Overview

```python
# number of rows and columns
loan_dataset.shape
# Output: (614, 13)
```

---

## 🔎 Data Preprocessing

```python
# label encoding
loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}}, inplace=True)

# dependent column values
loan_dataset['Dependents'].value_counts()
# Output:
# 0     274
# 2      85
# 1      80
# 3+     41
```

---

## 📈 Data Visualization

```python
# education & Loan Status
sns.countplot(x='Education', hue='Loan_Status', data=loan_dataset)

# marital status & Loan Status
sns.countplot(x='Married', hue='Loan_Status', data=loan_dataset)
```

---

## 🎯 Model Accuracy

### ✅ Training Data

```python
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data : ', training_data_accuracy)
# Output: 0.799 ✅
```

### ✅ Test Data

```python
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on test data : ', test_data_accuracy)
# Output: 0.833 ✅
```

---

## 🛠️ How to Run this Project

### 🔹 Clone the Repository

```bash
git clone https://github.com/your-username/loan-status-prediction.git
cd loan-status-prediction
```

### 🔹 Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔹 Run the Code

```bash
python main.py
```

---

## ⭐ Support

If you found this repo useful, **don’t forget to star ⭐ the repository!** 🚀

---
