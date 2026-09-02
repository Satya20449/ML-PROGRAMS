import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==================================================
# 1. LOAD DATASET
# ==================================================

def load_data(filename):

    # Read CSV file
    data = pd.read_csv(filename)

    print()
    print("========================================")
    print("DATASET COLUMNS")
    print("========================================")

    print(data.columns.tolist())

    print()
    print("========================================")
    print("FIRST 5 ROWS")
    print("========================================")

    print(data.head())


    # ----------------------------------------------
    # INPUT FEATURES
    # ----------------------------------------------

    features = [
        "CGPA",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore"
    ]


    # ----------------------------------------------
    # CHECK FEATURES
    # ----------------------------------------------

    missing_features = []

    for feature in features:

        if feature not in data.columns:

            missing_features.append(feature)


    if missing_features:

        print()
        print("ERROR!")
        print("The following feature columns are missing:")

        print(missing_features)

        print()
        print("Available columns are:")

        print(data.columns.tolist())

        raise ValueError(
            "Please check your dataset column names."
        )


    # ----------------------------------------------
    # AUTOMATICALLY FIND NUMERIC TARGET COLUMN
    # ----------------------------------------------

    possible_targets = [

        "PlacementPackage",
        "Package",
        "Salary",
        "CTC",
        "PlacementSalary",
        "ExpectedSalary",
        "AnnualSalary",
        "MonthlySalary"

    ]


    target = None


    # Check common target names

    for column in possible_targets:

        if column in data.columns:

            target = column

            break


    # If no common name is found,
    # find another numeric column

    if target is None:

        numeric_columns = data.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()


        remaining_numeric_columns = [

            column

            for column in numeric_columns

            if column not in features

        ]


        if len(remaining_numeric_columns) > 0:

            target = remaining_numeric_columns[0]


    # ----------------------------------------------
    # IF NO NUMERIC TARGET EXISTS
    # ----------------------------------------------

    if target is None:

        print()
        print("========================================")
        print("LINEAR REGRESSION CANNOT RUN")
        print("========================================")

        print()

        print(
            "Your dataset does not contain a suitable "
            "numeric target column."
        )

        print()

        print("Available columns:")

        print(data.columns.tolist())

        print()

        print(
            "Linear Regression requires a numeric value "
            "such as Salary, Package, or CTC."
        )

        raise ValueError(
            "No numeric target column found."
        )


    print()

    print(
        "Selected Target Column:",
        target
    )


    # ----------------------------------------------
    # SELECT DATA
    # ----------------------------------------------

    required_columns = features + [target]

    data = data[required_columns]


    # Convert everything to numeric

    for column in required_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )


    # Remove missing values

    data = data.dropna()


    # Input features

    X = data[features]


    # Target

    y = data[target]


    return X, y, target


# ==================================================
# 2. SPLIT DATASET
# ==================================================

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42

    )

    return X_train, X_test, y_train, y_test


# ==================================================
# 3. TRAIN AND EVALUATE MODEL
# ==================================================

def train_and_evaluate(

    X_train,
    X_test,

    y_train,
    y_test,

    name

):

    # Create model

    model = LinearRegression()


    # Train model

    model.fit(
        X_train,
        y_train
    )


    # Predictions

    train_pred = model.predict(
        X_train
    )


    test_pred = model.predict(
        X_test
    )


    # ----------------------------------------------
    # TRAINING METRICS
    # ----------------------------------------------

    train_mae = mean_absolute_error(
        y_train,
        train_pred
    )


    train_mse = mean_squared_error(
        y_train,
        train_pred
    )


    train_rmse = train_mse ** 0.5


    train_r2 = r2_score(
        y_train,
        train_pred
    )


    # ----------------------------------------------
    # TESTING METRICS
    # ----------------------------------------------

    test_mae = mean_absolute_error(
        y_test,
        test_pred
    )


    test_mse = mean_squared_error(
        y_test,
        test_pred
    )


    test_rmse = test_mse ** 0.5


    test_r2 = r2_score(
        y_test,
        test_pred
    )


    # ----------------------------------------------
    # DISPLAY RESULTS
    # ----------------------------------------------

    print()

    print("========================================")

    print(name)

    print("========================================")


    print()

    print("TRAINING RESULTS")

    print("-----------------------------")

    print(
        "MAE:",
        round(train_mae, 4)
    )

    print(
        "MSE:",
        round(train_mse, 4)
    )

    print(
        "RMSE:",
        round(train_rmse, 4)
    )

    print(
        "R2 Score:",
        round(train_r2, 4)
    )


    print()

    print("TESTING RESULTS")

    print("-----------------------------")

    print(
        "MAE:",
        round(test_mae, 4)
    )

    print(
        "MSE:",
        round(test_mse, 4)
    )

    print(
        "RMSE:",
        round(test_rmse, 4)
    )

    print(
        "R2 Score:",
        round(test_r2, 4)
    )


    return model, test_r2


# ==================================================
# 4. MAIN PROGRAM
# ==================================================

def main():

    # ----------------------------------------------
    # CSV FILE PATH
    # ----------------------------------------------

    filename = r"C:\Users\ASUS\Downloads\ML PROGRAMS - Copy\placementprediction.csv"


    # ----------------------------------------------
    # LOAD DATA
    # ----------------------------------------------

    X, y, target_name = load_data(
        filename
    )


    print()

    print("========================================")

    print("LINEAR REGRESSION MODEL")

    print("========================================")


    print()

    print(
        "Dataset loaded successfully!"
    )


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

        print(
            "-",
            feature
        )


    print()

    print(
        "Target:",
        target_name
    )


    # ----------------------------------------------
    # SPLIT DATA
    # ----------------------------------------------

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


    # ==================================================
    # 5. UNSCALED LINEAR REGRESSION
    # ==================================================

    model_unscaled, r2_unscaled = train_and_evaluate(

        X_train,
        X_test,

        y_train,
        y_test,

        "Unscaled Linear Regression"

    )


    # ==================================================
    # 6. STANDARD SCALER
    # ==================================================

    standard_scaler = StandardScaler()


    X_train_standard = standard_scaler.fit_transform(
        X_train
    )


    X_test_standard = standard_scaler.transform(
        X_test
    )


    model_standard, r2_standard = train_and_evaluate(

        X_train_standard,
        X_test_standard,

        y_train,
        y_test,

        "StandardScaler Linear Regression"

    )


    # ==================================================
    # 7. MINMAX SCALER
    # ==================================================

    minmax_scaler = MinMaxScaler()


    X_train_minmax = minmax_scaler.fit_transform(
        X_train
    )


    X_test_minmax = minmax_scaler.transform(
        X_test
    )


    model_minmax, r2_minmax = train_and_evaluate(

        X_train_minmax,
        X_test_minmax,

        y_train,
        y_test,

        "MinMaxScaler Linear Regression"

    )


    # ==================================================
    # 8. ACTUAL VS PREDICTED GRAPH
    # ==================================================

    y_pred = model_standard.predict(
        X_test_standard
    )


    plt.figure()


    plt.scatter(
        y_test,
        y_pred
    )


    plt.xlabel(
        "Actual " + target_name
    )


    plt.ylabel(
        "Predicted " + target_name
    )


    plt.title(
        "Actual vs Predicted - Linear Regression"
    )


    plt.grid(True)


    plt.tight_layout()


    plt.show()


    # ==================================================
    # 9. NEW STUDENT PREDICTION
    # ==================================================

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

            columns=[

                "CGPA",

                "AptitudeTestScore",

                "CodingTestScore",

                "MockInterviewScore"

            ]

        )


        new_student_scaled = standard_scaler.transform(
            new_student
        )


        predicted_value = model_standard.predict(
            new_student_scaled
        )[0]


        print()

        print(
            "Predicted",
            target_name + ":",
            round(predicted_value, 2)
        )


    except ValueError:

        print()

        print(
            "Invalid input! Please enter numeric values."
        )


    # ==================================================
    # 10. FINAL MODEL COMPARISON
    # ==================================================

    print()

    print("========================================")

    print("MODEL COMPARISON")

    print("========================================")


    print()

    print(
        "Unscaled R2 Score:",
        round(r2_unscaled, 4)
    )


    print(
        "StandardScaler R2 Score:",
        round(r2_standard, 4)
    )


    print(
        "MinMaxScaler R2 Score:",
        round(r2_minmax, 4)
    )


    # ==================================================
    # 11. PROGRAM COMPLETE
    # ==================================================

    print()

    print("========================================")

    print("MODEL COMPLETE")

    print("========================================")


    print()

    print(
        "Linear Regression evaluation completed successfully!"
    )


# ==================================================
# PROGRAM ENTRY POINT
# ==================================================

if __name__ == "__main__":

    main()