import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

np.random.seed(42)

# ------------------------------
# 1) Synthetic dataset creation
# ------------------------------
n_samples = 1000

monthly_income = np.random.randint(15000, 150000, n_samples)
age = np.random.randint(21, 60, n_samples)
work_experience = np.random.randint(0, 30, n_samples)
existing_emi = np.random.randint(0, 30000, n_samples)
credit_score = np.random.randint(550, 850, n_samples)
loan_amount = np.random.randint(100000, 3000000, n_samples)
tenure_years = np.random.randint(1, 20, n_samples)

interest_rate = 0.12  
tenure_months = tenure_years * 12
monthly_rate = interest_rate / 12

emi = (loan_amount * monthly_rate * (1 + monthly_rate) ** tenure_months) / \
      ((1 + monthly_rate) ** tenure_months - 1)
emi = emi + np.random.normal(0, 1000, n_samples)

risk_score = (
    0.4 * (existing_emi / (monthly_income + 1)) * 100 +
    0.3 * (loan_amount / (monthly_income * 12 + 1)) * 10 -
    0.3 * ((credit_score - 550) / 300) * 100
)

approval = np.where(
    (risk_score < 40) &
    (credit_score > 650) &
    (monthly_income > 25000),
    1, 0
)

df = pd.DataFrame({
    "monthly_income": monthly_income,
    "age": age,
    "work_experience": work_experience,
    "existing_emi": existing_emi,
    "credit_score": credit_score,
    "loan_amount": loan_amount,
    "tenure_years": tenure_years,
    "emi": emi,
    "approval": approval,
    "risk_score": risk_score
})

# ------------------------------
# 2) EMI Prediction Model
# ------------------------------
X_emi = df[["monthly_income", "existing_emi", "loan_amount", "tenure_years", "credit_score"]]
y_emi = df["emi"]

X_train, X_test, y_train, y_test = train_test_split(X_emi, y_emi, test_size=0.2, random_state=42)

emi_model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

emi_model.fit(X_train, y_train)

joblib.dump(emi_model, "models/emi_model.pkl")
print("EMI model saved.")

# ------------------------------
# 3) Approval Prediction Model
# ------------------------------
X_app = df[["monthly_income", "existing_emi", "loan_amount", "tenure_years", "credit_score", "risk_score"]]
y_app = df["approval"]

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_app, y_app, test_size=0.2, random_state=42)

approval_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])

approval_model.fit(X_train2, y_train2)

joblib.dump(approval_model, "models/approval_model.pkl")
print("Approval model saved.")
