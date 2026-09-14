# ============================================================
# BAGGING CLASSIFIER - PLACEMENT PREDICTION
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)


# ============================================================
# PATH
# ============================================================

DATA_FILE = r"C:\Users\ASUS\Downloads\ML PROGRAMS - Copy\placement_preprocessed.csv"


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

TARGET = "PlacementStatus"

if TARGET not in data.columns:

    print("\nERROR: PlacementStatus column not found.")

    raise SystemExit


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = data.drop(
    columns=[
        "PlacementStatus",
        "Salary Package",
        "IsAnomaly"
    ],
    errors="ignore"
)

y = data[TARGET]


# ============================================================
# CONVERT TO NUMERICAL
# ============================================================

X = X.apply(pd.to_numeric, errors="coerce")

y = pd.to_numeric(
    y,
    errors="coerce"
)


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

valid_rows = (
    X.notnull().all(axis=1)
    & y.notnull()
)

X = X[valid_rows]

y = y[valid_rows]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data :", X_test.shape)


# ============================================================
# BASE DECISION TREE
# ============================================================

base_model = DecisionTreeClassifier(
    random_state=42
)


# ============================================================
# BAGGING MODEL
# ============================================================

model = BaggingClassifier(
    estimator=base_model,
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Bagging Classifier...")

model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    y_probability
)


# ============================================================
# RESULTS
# ============================================================

print("\n============================================")
print("BAGGING CLASSIFIER RESULTS")
print("============================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Probability": y_probability
})

results["Probability"] = results["Probability"].round(4)

print("\nSample Predictions:")
print(results.head(10))


# ============================================================
# CONFUSION MATRIX GRAPH
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Confusion Matrix - Bagging Classifier"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.colorbar()

plt.xticks(
    [0, 1],
    ["Not Placed", "Placed"]
)

plt.yticks(
    [0, 1],
    ["Not Placed", "Placed"]
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.show()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Bagging (AUC = {auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - Bagging Classifier"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n============================================")
print("BAGGING CLASSIFIER COMPLETED")
print("============================================")