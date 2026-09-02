import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve
)


# ==================================================
# 1. LOAD DATASET
# ==================================================

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
            "Missing columns: " + str(missing_columns)
        )

    data = data[required_columns].dropna()

    X = data[features]
    y = data[target]

    return X, y


# ==================================================
# 2. SPLIT DATASET
# ==================================================

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# ==================================================
# 3. TRAIN ADABOOST MODEL
# ==================================================

def train_model(X_train, y_train):

    model = AdaBoostClassifier(
        n_estimators=100,
        learning_rate=1.0,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


# ==================================================
# 4. EVALUATE MODEL
# ==================================================

def evaluate_model(model, X_train, X_test, y_train, y_test):

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(
        y_train,
        train_pred
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )

    print()
    print("========================================")
    print("ADABOOST RESULTS")
    print("========================================")

    print("Training Accuracy:",
          round(train_accuracy, 4))

    print("Testing Accuracy :",
          round(test_accuracy, 4))

    print()
    print("========================================")
    print("CLASSIFICATION REPORT")
    print("========================================")

    print(
        classification_report(
            y_test,
            test_pred
        )
    )

    return test_pred, train_accuracy, test_accuracy


# ==================================================
# 5. CONFUSION MATRIX
# ==================================================

def show_confusion_matrix(model, X_test, y_test):

    y_pred = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print()
    print("========================================")
    print("CONFUSION MATRIX")
    print("========================================")

    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_
    )

    display.plot()

    plt.title(
        "AdaBoost - Confusion Matrix"
    )

    plt.tight_layout()

    plt.show()


# ==================================================
# 6. FEATURE IMPORTANCE
# ==================================================

def show_feature_importance(model, feature_names):

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print()
    print("========================================")
    print("FEATURE IMPORTANCE")
    print("========================================")

    print(importance)

    plt.figure(figsize=(8, 5))

    plt.bar(
        importance["Feature"],
        importance["Importance"]
    )

    plt.xlabel("Features")
    plt.ylabel("Importance")

    plt.title(
        "AdaBoost Feature Importance"
    )

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.show()


# ==================================================
# 7. ROC-AUC AND ROC CURVE
# ==================================================

def show_roc_auc(model, X_test, y_test):

    print()
    print("========================================")
    print("ROC-AUC")
    print("========================================")

    if len(model.classes_) != 2:

        print(
            "ROC-AUC is only available for "
            "binary classification."
        )

        return

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    positive_class = model.classes_[1]

    y_test_binary = (
        y_test == positive_class
    ).astype(int)

    auc_score = roc_auc_score(
        y_test_binary,
        y_probability
    )

    print(
        "ROC-AUC Score:",
        round(auc_score, 4)
    )

    fpr, tpr, thresholds = roc_curve(
        y_test_binary,
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

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve - AdaBoost"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()


# ==================================================
# 8. NEW STUDENT PREDICTION
# ==================================================

def predict_new_student(model, feature_names):

    print()
    print("========================================")
    print("NEW STUDENT PREDICTION")
    print("========================================")

    try:

        cgpa = float(
            input("Enter CGPA: ")
        )

        aptitude = float(
            input("Enter Aptitude Test Score: ")
        )

        coding = float(
            input("Enter Coding Test Score: ")
        )

        mock_interview = float(
            input("Enter Mock Interview Score: ")
        )

        new_student = pd.DataFrame(
            [[
                cgpa,
                aptitude,
                coding,
                mock_interview
            ]],
            columns=feature_names
        )

        prediction = model.predict(
            new_student
        )[0]

        probabilities = model.predict_proba(
            new_student
        )[0]

        print()
        print(
            "Predicted Placement Status:",
            prediction
        )

        print()
        print("Class Probabilities:")

        for class_name, probability in zip(
            model.classes_,
            probabilities
        ):
            print(
                class_name,
                ":",
                round(probability * 100, 2),
                "%"
            )

    except ValueError:

        print(
            "Invalid input! Please enter numeric values."
        )


# ==================================================
# 9. MAIN PROGRAM
# ==================================================

def main():

    filename = r"C:\Users\ASUS\Downloads\ML PROGRAMS - Copy\placementprediction.csv"

    print("========================================")
    print("PLACEMENT PREDICTION PROJECT")
    print("ADABOOST CLASSIFIER")
    print("========================================")

    X, y = load_data(filename)

    print()
    print("Dataset loaded successfully!")

    print(
        "Number of rows:",
        len(X)
    )

    print(
        "Number of features:",
        X.shape[1]
    )

    print()
    print("Features:")

    for feature in X.columns:
        print("-", feature)

    print()
    print("Target:", y.name)

    print()
    print("Target Classes:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    print()
    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )

    model = train_model(
        X_train,
        y_train
    )

    print()
    print(
        "AdaBoost model trained successfully!"
    )

    y_pred, train_accuracy, test_accuracy = (
        evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    show_confusion_matrix(
        model,
        X_test,
        y_test
    )

    show_feature_importance(
        model,
        X.columns.tolist()
    )

    show_roc_auc(
        model,
        X_test,
        y_test
    )

    predict_new_student(
        model,
        X.columns.tolist()
    )

    print()
    print("========================================")
    print("MODEL COMPLETE")
    print("========================================")

    print(
        "Training Accuracy:",
        round(train_accuracy, 4)
    )

    print(
        "Testing Accuracy:",
        round(test_accuracy, 4)
    )

    print()
    print(
        "AdaBoost implementation completed successfully!"
    )


# ==================================================
# PROGRAM ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()