import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
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
        print("\nAvailable columns:")
        print(data.columns.tolist())

        raise ValueError(
            f"\nMissing columns: {missing_columns}"
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
# 3. TRAIN DECISION TREE
# ==================================================

def train_model(X_train, y_train):

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
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

    print("\n========================================")
    print("DECISION TREE RESULTS")
    print("========================================")

    print("\nTraining Accuracy:",
          round(train_accuracy, 4))

    print("Testing Accuracy :",
          round(test_accuracy, 4))

    print("\n========================================")
    print("CLASSIFICATION REPORT")
    print("========================================")

    print(
        classification_report(
            y_test,
            test_pred
        )
    )

    return test_pred


# ==================================================
# 5. CONFUSION MATRIX
# ==================================================

def show_confusion_matrix(y_test, y_pred):

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\n========================================")
    print("CONFUSION MATRIX")
    print("========================================")

    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    display.plot()

    plt.title(
        "Decision Tree - Confusion Matrix"
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

    print("\n========================================")
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
        "Decision Tree Feature Importance"
    )

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.show()


# ==================================================
# 7. VISUALIZE DECISION TREE
# ==================================================

def visualize_tree(model, feature_names, class_names):

    plt.figure(figsize=(20, 10))

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=[str(x) for x in class_names],
        filled=True,
        rounded=True,
        fontsize=8
    )

    plt.title(
        "Placement Prediction Decision Tree"
    )

    plt.tight_layout()

    plt.show()


# ==================================================
# 8. NEW STUDENT PREDICTION
# ==================================================

def predict_new_student(model, feature_names):

    print("\n========================================")
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

        print("\nPredicted Placement Status:",
              prediction)

    except ValueError:

        print(
            "\nInvalid input! Please enter numeric values."
        )


# ==================================================
# 9. MAIN PROGRAM
# ==================================================

def main():

    filename = r"C:\Users\ASUS\Downloads\ML PROGRAMS - Copy\placementprediction.csv"

    print("========================================")
    print("PLACEMENT PREDICTION PROJECT")
    print("DECISION TREE CLASSIFIER")
    print("========================================")

    X, y = load_data(filename)

    print("\nDataset loaded successfully!")

    print("Number of rows:",
          len(X))

    print("Number of features:",
          X.shape[1])

    print("\nFeatures:")

    for feature in X.columns:
        print("-", feature)

    print("\nTarget:",
          y.name)

    print("\nClasses:",
          y.unique())

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    print("\nTraining samples:",
          len(X_train))

    print("Testing samples :",
          len(X_test))

    model = train_model(
        X_train,
        y_train
    )

    print("\nDecision Tree trained successfully!")

    y_pred = evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    show_confusion_matrix(
        y_test,
        y_pred
    )

    show_feature_importance(
        model,
        X.columns
    )

    visualize_tree(
        model,
        X.columns,
        model.classes_
    )

    predict_new_student(
        model,
        X.columns
    )

    print("\n========================================")
    print("MODEL COMPLETE")
    print("========================================")

    print("\nDecision Tree implementation completed successfully!")


# ==================================================
# PROGRAM ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()