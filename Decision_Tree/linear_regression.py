# ============================================================
# LINEAR REGRESSION - PLACEMENT DATASET
# Predict Salary Package
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "placement_preprocessed.csv"


# ============================================================
# READ DATASET
# ============================================================

print("Reading preprocessed dataset...")

data = pd.read_csv(DATA_FILE)

print("\nDataset Shape:")
print(data.shape)

print("\nFirst 5 Rows:")
print(data.head())


# ============================================================
# TARGET
# ============================================================

TARGET = "Salary Package"


if TARGET not in data.columns:

    print("\nERROR:")
    print("Salary Package column not found.")
    print("Please run preprocessing.py again.")

    raise SystemExit


# ============================================================
# REMOVE TARGET FROM FEATURES
# ============================================================

X = data.drop(
    columns=[
        "Salary Package",
        "PlacementStatus"
    ],
    errors="ignore"
)

y = data[TARGET]


# ============================================================
# REMOVE INVALID VALUES
# ============================================================

X = X.apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(y, errors="coerce")

valid_rows = X.notnull().all(axis=1) & y.notnull()

X = X[valid_rows]
y = y[valid_rows]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining Data:", X_train.shape)
print("Testing Data :", X_test.shape)


# ============================================================
# CREATE MODEL
# ============================================================

model = LinearRegression()


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Linear Regression model...")

model.fit(X_train, y_train)


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)


print("\n============================================")
print("LINEAR REGRESSION RESULTS")
print("============================================")

print(f"Mean Absolute Error : {mae:.4f}")
print(f"Mean Squared Error  : {mse:.4f}")
print(f"Root Mean Squared Error : {rmse:.4f}")
print(f"R2 Score            : {r2:.4f}")


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Actual Salary": y_test.values,
    "Predicted Salary": y_pred
})

results["Predicted Salary"] = results["Predicted Salary"].round(2)

print("\nSample Predictions:")
print(results.head(10))


# ============================================================
# ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

plt.xlabel("Actual Salary Package")
plt.ylabel("Predicted Salary Package")

plt.title(
    "Linear Regression - Actual vs Predicted Salary"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# REGRESSION LINE
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    y_test.values,
    label="Actual Salary"
)

plt.plot(
    y_pred,
    label="Predicted Salary"
)

plt.xlabel("Test Sample")

plt.ylabel("Salary Package")

plt.title(
    "Linear Regression - Actual vs Predicted"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


print("\n============================================")
print("LINEAR REGRESSION COMPLETED")
print("============================================")