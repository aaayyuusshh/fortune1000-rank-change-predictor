# Fortune 1000 Rank Change Predictor 🏢📊

A machine learning project built with **Python** and **TensorFlow** to predict whether Fortune 1000 companies changed their rank from the previous year, based on their financial and operational metrics.

---

## 📌 Project Overview
This project explores the Fortune Top 1000 Companies by Revenue (2022) dataset and applies a **binary classification neural network** to predict if a company’s rank changed year-over-year (`change_in_rank`: 1 = changed, 0 = same).

Key goals:
- Perform **data cleaning** and preprocessing
- Conduct **exploratory data analysis (EDA)**
- Build, train, and evaluate a **TensorFlow-based machine learning model**
- Address **overfitting** using dropout and hyperparameter tuning

---

## 📂 Dataset
- **Source:** [Kaggle — Fortune Top 1000 Companies by Revenue 2022](https://www.kaggle.com/datasets/surajjha101/fortune-top-1000-companies-by-revenue-2022)
- **Size:** 1000 rows × 10 columns  
- **Features used (inputs):**
  - `revenue`
  - `revenue_percent_change`
  - `profit`
  - `profit_percent_change`
  - `assets`
  - `market_value`
  - `employees`
- **Target (output):**
  - `change_in_rank` (1 = changed, 0 = same)

---

## 🧹 Data Cleaning
Performed using **pandas** and **numpy**:
- Removed rows with missing or invalid entries (`-`)
- Stripped `$`, `%`, and `()` characters from numeric columns
- Converted all numeric data from string → `int/float`
- Encoded non-numeric categorical values into binary (0/1)
- Split dataset into **70% train / 30% test**

---

## 📊 Exploratory Data Analysis (EDA)
Performed using **matplotlib**:
- Bar chart of top 15 companies by revenue
- Scatter plots showing:
  - Revenue vs Market Value
  - Employees vs Profit

Key insights:
- Higher revenue ≠ always higher market value
- More employees ≠ always higher profit

Visualizatons can be found under `/visualizations`

---

## 🤖 Model Architecture
Built with **TensorFlow / Keras**:

- **Hidden Layers:**
  - 2 layers, 128 neurons each, `ReLU` activation
- **Output Layer:**
  - 1 neuron, `sigmoid` activation
- **Optimizer:** `Adam`
- **Loss Function:** `binary_crossentropy`
- **Epochs:** 200  
- **Regularization:** `Dropout(0.8)` to reduce overfitting

---

## Sample use of this model in code
```python
import tensorflow as tf
import numpy as np

# Load our model
model = tf.keras.models.load_model("model.h5")

# Example input: [revenue, revenue_pct_change, profit, profit_pct_change, assets, market_value, employees]
sample_company = np.array([[120_000, 5.3, 10_000, 2.1, 300_000, 150_000, 50000]], dtype=float)

# Predict probability of rank change
probability = model.predict(sample_company)[0][0]
prediction = 1 if probability >= 0.5 else 0

print(f"Predicted probability of rank change: {probability:.2%}")
print("Predicted label:", "Changed" if prediction == 1 else "Same")
```

---

˖☆ ࣪⭑ ݁˖ .☆ . ݁ ˖  ☆⊹ ࣪  ˖ ࣪　⊹ ࣪ ★ ⋆.˚  ⊹ ࣪
