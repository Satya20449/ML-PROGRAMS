import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)


def load_data(filename):
    data = pd.read_csv(filename)

    features = [
        "CGPA",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore"
    ]

    target = "PlacementStatus"

    required_columns = features + [target]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        print("Available columns:", data.columns.tolist())
        raise ValueError(
            "Missing columns in dataset: " + str(missing_columns)
        )

    data = data[required_columns].dropna()

    X = data[features]
    y = data[target]

    return X, y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_and_evaluate(
    X_train,
    X_test,
    y_train,
    y_test,
    name
):
    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    print()
    print("========================================")
    print(name)
    print("========================================")
    print("Train Accuracy:", round(train_accuracy, 4))
    print("Test Accuracy :", round(test_accuracy, 4))

    return model


def main():

    filename = r"C:\Users\ASUS\Downloads\ML PROGRAMS - Copy\placementprediction.csv"

    X, y = load_data(filename)

    print("========================================")
    print("PLACEMENT PREDICTION MODEL")
    print("LOGISTIC REGRESSION")
    print("========================================")

    print()
    print("Dataset loaded successfully!")
    print("Number of rows:", len(X))
    print("Number of features:", X.shape[1])
    print("Features:", X.columns.tolist())
    print("Target:", y.name)
    print("Classes:", y.unique())

    X_train, X_test, y_train, y_test = split_data(X, y)

    print()
    print("Training samples:", len(X_train))
    print("Testing samples :", len(X_test))

    model_unscaled = train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
        "Unscaled Logistic Regression"
    )

    standard_scaler = StandardScaler()

    X_train_standard = standard_scaler.fit_transform(X_train)
    X_test_standard = standard_scaler.transform(X_test)

    model_standard = train_and_evaluate(
        X_train_standard,
        X_test_standard,
        y_train,
        y_test,
        "StandardScaler Logistic Regression"
    )

    minmax_scaler = MinMaxScaler()

    X_train_minmax = minmax_scaler.fit_transform(X_train)
    X_test_minmax = minmax_scaler.transform(X_test)

    model_minmax = train_and_evaluate(
        X_train_minmax,
        X_test_minmax,
        y_train,
        y_test,
        "MinMaxScaler Logistic Regression"
    )

    y_pred = model_standard.predict(X_test_standard)

    cm = confusion_matrix(y_test, y_pred)

    print()
    print("========================================")
    print("CONFUSION MATRIX")
    print("========================================")
    print(cm)

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model_standard.classes_
    ).plot()

    plt.title("Confusion Matrix - StandardScaler")
    plt.tight_layout()
    plt.show()

    print()
    print("========================================")
    print("ROC-AUC")
    print("========================================")

    if len(model_standard.classes_) == 2:
        label_encoder = LabelEncoder()

        y_test_encoded = label_encoder.fit_transform(y_test)

        y_probability = model_standard.predict_proba(
            X_test_standard
        )[:, 1]

        auc_score = roc_auc_score(
            y_test_encoded,
            y_probability
        )

        print("ROC-AUC Score:", round(auc_score, 4))

        fpr, tpr, thresholds = roc_curve(
            y_test_encoded,
            y_probability
        )

        plt.figure()

        plt.plot(
            fpr,
            tpr,
            label=f"ROC Curve (AUC = {auc_score:.4f})"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Random Classifier"
        )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - StandardScaler Logistic Regression")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    else:
        print("ROC-AUC is only shown for binary classification.")

    print()
    print("========================================")
    print("MODEL COMPLETE")
    print("========================================")
    print("Models tested:")
    print("1. Unscaled Logistic Regression")
    print("2. StandardScaler Logistic Regression")
    print("3. MinMaxScaler Logistic Regression")


if __name__ == "__main__":
    main()
