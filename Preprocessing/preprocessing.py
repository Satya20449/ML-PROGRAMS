# ============================================================
# PLACEMENT DATASET - COMPLETE PREPROCESSING
# ============================================================

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR.parent / "placementprediction.csv"
OUTPUT_FILE = BASE_DIR.parent / "placement_preprocessed.csv"


# ============================================================
# READ DATASET
# ============================================================

print("Reading dataset...")

data = pd.read_csv(INPUT_FILE)

print("\nOriginal Dataset:")
print(data.head())

print("\nOriginal Shape:", data.shape)


# ============================================================
# REMOVE UNNECESSARY COLUMNS
# ============================================================

# Remove unwanted index column if present
if "Unnamed: 0" in data.columns:
    data = data.drop(columns=["Unnamed: 0"])

# StudentID is only an identifier
if "StudentID" in data.columns:
    data = data.drop(columns=["StudentID"])


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

numerical_columns = data.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = data.select_dtypes(
    include=["object", "string"]
).columns.tolist()


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

print("\nHandling missing values...")


# Numerical columns → median
for column in numerical_columns:
    data[column] = data[column].fillna(
        data[column].median()
    )


# Categorical columns → mode
for column in categorical_columns:

    if not data[column].mode().empty:

        data[column] = data[column].fillna(
            data[column].mode()[0]
        )

    else:

        data[column] = data[column].fillna("Unknown")


# ============================================================
# REMOVE NUMERICAL OUTLIERS
# ============================================================

print("\nRemoving numerical outliers...")


# Do NOT remove rows based on target/output columns
excluded_from_outlier = [
    "PlacementStatus",
    "Salary Package",
    "IsAnomaly"
]


outlier_columns = [
    column
    for column in numerical_columns
    if column not in excluded_from_outlier
]


if len(outlier_columns) > 0:

    Q1 = data[outlier_columns].quantile(0.25)

    Q3 = data[outlier_columns].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR

    upper_bound = Q3 + 1.5 * IQR


    mask = ~(
        (
            data[outlier_columns] < lower_bound
        )
        |
        (
            data[outlier_columns] > upper_bound
        )
    ).any(axis=1)


    data = data[mask].reset_index(drop=True)


print(
    "Shape after outlier removal:",
    data.shape
)


# ============================================================
# IDENTIFY COLUMNS AGAIN
# ============================================================

numerical_columns = data.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = data.select_dtypes(
    include=["object", "string"]
).columns.tolist()


# ============================================================
# FEATURES
# ============================================================

# These are targets/output columns.
# They must NOT be used as input features.

target_columns = [
    "PlacementStatus",
    "Salary Package"
]


# Numerical features
feature_numerical_columns = [
    column
    for column in numerical_columns
    if column not in target_columns
]


# ============================================================
# DISPLAY COLUMNS
# ============================================================

print("\nNumerical Feature Columns:")
print(feature_numerical_columns)

print("\nCategorical Columns:")
print(categorical_columns)

print("\nTarget Columns:")

if "PlacementStatus" in data.columns:
    print("- PlacementStatus")

if "Salary Package" in data.columns:
    print("- Salary Package")


# ============================================================
# NORMALIZATION
# ============================================================

print("\nNormalizing numerical columns...")


scaled_data = pd.DataFrame(
    index=data.index
)


if len(feature_numerical_columns) > 0:

    scaler = MinMaxScaler()

    scaled_values = scaler.fit_transform(
        data[feature_numerical_columns]
    )


    scaled_data = pd.DataFrame(
        scaled_values,
        columns=[
            "numerical__" + column
            for column in feature_numerical_columns
        ],
        index=data.index
    )


# ============================================================
# CATEGORICAL ENCODING
# ============================================================

print(
    "\nConverting categorical columns into numerical values..."
)


encoded_data = pd.DataFrame(
    index=data.index
)


if len(categorical_columns) > 0:

    # Works with newer and older scikit-learn
    try:

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

    except TypeError:

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


    encoded_values = encoder.fit_transform(
        data[categorical_columns]
    )


    encoded_columns = encoder.get_feature_names_out(
        categorical_columns
    )


    encoded_data = pd.DataFrame(
        encoded_values,
        columns=[
            "categorical__" + column
            for column in encoded_columns
        ],
        index=data.index
    )


# ============================================================
# KEEP ISANOMALY AS NUMERICAL FEATURE
# ============================================================

anomaly_data = pd.DataFrame(
    index=data.index
)


if "IsAnomaly" in data.columns:

    anomaly_data["IsAnomaly"] = pd.to_numeric(
        data["IsAnomaly"],
        errors="coerce"
    )


# ============================================================
# KEEP TARGET COLUMNS
# ============================================================

target_data = pd.DataFrame(
    index=data.index
)


# PlacementStatus → classification target
if "PlacementStatus" in data.columns:

    target_data["PlacementStatus"] = pd.to_numeric(
        data["PlacementStatus"],
        errors="coerce"
    )


# Salary Package → regression target
if "Salary Package" in data.columns:

    target_data["Salary Package"] = pd.to_numeric(
        data["Salary Package"],
        errors="coerce"
    )


# ============================================================
# COMBINE ALL PROCESSED DATA
# ============================================================

print("\nCombining processed data...")


final_data = pd.concat(
    [
        scaled_data,
        encoded_data,
        anomaly_data,
        target_data
    ],
    axis=1
)


# ============================================================
# MAKE SURE EVERYTHING IS NUMERICAL
# ============================================================

final_data = final_data.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

final_data = final_data.dropna(
    axis=0
).reset_index(drop=True)


# ============================================================
# ROUND DECIMAL VALUES
# ============================================================

final_data = final_data.round(4)


# ============================================================
# SAVE PREPROCESSED DATASET
# ============================================================

final_data.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n============================================")
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("============================================")


print("\nOutput File:")
print(OUTPUT_FILE)


print("\nFinal Shape:")
print(final_data.shape)


print("\nFinal Dataset:")
print(final_data.head())


# ============================================================
# CHECK NON-NUMERICAL COLUMNS
# ============================================================

print(
    "\nChecking for non-numerical columns..."
)


non_numeric_columns = final_data.select_dtypes(
    exclude=["number"]
).columns.tolist()


if len(non_numeric_columns) == 0:

    print(
        "SUCCESS: All columns contain only numerical values."
    )

else:

    print(
        "WARNING: Non-numerical columns found:"
    )

    print(non_numeric_columns)


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\nChecking missing values...")


missing_values = final_data.isnull().sum().sum()


if missing_values == 0:

    print(
        "SUCCESS: No missing values."
    )

else:

    print(
        "WARNING: Missing values found:",
        missing_values
    )


# ============================================================
# CHECK TARGET COLUMNS
# ============================================================

print("\nTarget columns in final dataset:")


if "PlacementStatus" in final_data.columns:

    print(
        "PlacementStatus: Present"
    )


if "Salary Package" in final_data.columns:

    print(
        "Salary Package: Present"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n============================================")
print("placement_preprocessed.csv UPDATED")
print("============================================")