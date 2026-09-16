from flask import Flask, render_template
import pandas as pd
from pathlib import Path
import shutil


app = Flask(__name__)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Original dataset
DATA_FILE = BASE_DIR.parent / "placementprediction.csv"

# Preprocessed dataset used by ML models
PREPROCESSED_FILE = BASE_DIR.parent / "placement_preprocessed.csv"

# EDA output folder
EDA_SOURCE = BASE_DIR.parent / "eda_output"

# Dashboard EDA images folder
EDA_DESTINATION = BASE_DIR / "static" / "eda_images"

# Clustering output images folder
CLUSTERING_DESTINATION = BASE_DIR / "static" / "clustering_images"


# ============================================================
# COPY EDA IMAGES
# ============================================================

def copy_eda_images():

    # Create destination folder
    EDA_DESTINATION.mkdir(
        parents=True,
        exist_ok=True
    )

    # Copy all PNG files from EDA output
    if EDA_SOURCE.exists():

        for image in EDA_SOURCE.glob("*.png"):

            destination = EDA_DESTINATION / image.name

            try:

                shutil.copy2(
                    image,
                    destination
                )

            except Exception as e:

                print(
                    f"Could not copy {image.name}: {e}"
                )


# ============================================================
# COPY CLUSTERING IMAGES
# ============================================================

def copy_clustering_images():

    # Create clustering images folder
    CLUSTERING_DESTINATION.mkdir(
        parents=True,
        exist_ok=True
    )

    # Clustering image filenames
    clustering_files = [
        "kmeans_elbow_method.png",
        "kmeans_final_clusters.png",
        "kmeans_outliers.png",
        "kmeans_simple_clusters.png"
    ]

    # Copy each clustering image
    for filename in clustering_files:

        source = BASE_DIR.parent / filename
        destination = CLUSTERING_DESTINATION / filename

        if source.exists():

            try:

                shutil.copy2(
                    source,
                    destination
                )

                print(
                    f"Copied clustering image: {filename}"
                )

            except Exception as e:

                print(
                    f"Could not copy {filename}: {e}"
                )

        else:

            print(
                f"Clustering image not found: {filename}"
            )


# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

def load_data():

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Original dataset not found:\n{DATA_FILE}"
        )

    return pd.read_csv(DATA_FILE)


# ============================================================
# CHECK PREPROCESSED DATASET
# ============================================================

def check_preprocessed_data():

    if PREPROCESSED_FILE.exists():

        try:

            df = pd.read_csv(
                PREPROCESSED_FILE,
                nrows=5
            )

            return {
                "available": True,
                "columns": len(df.columns)
            }

        except Exception as e:

            print(
                f"Could not read preprocessed dataset: {e}"
            )

            return {
                "available": False,
                "columns": 0
            }

    return {
        "available": False,
        "columns": 0
    }


# ============================================================
# GET CLUSTERING IMAGES
# ============================================================

def get_clustering_images():

    clustering_images = []

    if CLUSTERING_DESTINATION.exists():

        clustering_images = sorted(
            [
                image.name
                for image
                in CLUSTERING_DESTINATION.glob("*.png")
            ]
        )

    return clustering_images


# ============================================================
# DASHBOARD HOME
# ============================================================

@app.route("/")
def home():

    print("\n============================================")
    print("Loading Placement Dashboard...")
    print("============================================")


    # ========================================================
    # COPY EDA IMAGES
    # ========================================================

    copy_eda_images()


    # ========================================================
    # COPY CLUSTERING IMAGES
    # ========================================================

    copy_clustering_images()


    # ========================================================
    # LOAD ORIGINAL DATASET
    # ========================================================

    df = load_data()


    # ========================================================
    # BASIC STATISTICS
    # ========================================================

    total_students = len(df)

    total_features = len(df.columns)


    # ========================================================
    # DATASET COLUMN NAMES
    # ========================================================

    column_names = df.columns.tolist()


    # ========================================================
    # MISSING VALUE REPORT
    # ========================================================

    missing_report = [

        {
            "Column": column,
            "Missing Values": int(
                df[column].isnull().sum()
            )
        }

        for column in df.columns

    ]


    # ========================================================
    # PLACEMENT STATUS
    # ========================================================

    if "PlacementStatus" in df.columns:

        placed_students = int(
            (
                df["PlacementStatus"] == 1
            ).sum()
        )

        not_placed_students = int(
            (
                df["PlacementStatus"] == 0
            ).sum()
        )

    else:

        placed_students = 0

        not_placed_students = 0


    # ========================================================
    # MISSING VALUES
    # ========================================================

    missing_values = int(
        df.isnull().sum().sum()
    )


    # ========================================================
    # DUPLICATES
    # ========================================================

    duplicates = int(
        df.duplicated().sum()
    )


    # ========================================================
    # PLACEMENT RATE
    # ========================================================

    if total_students > 0:

        placement_rate = round(

            (
                placed_students
                /
                total_students
            )
            * 100,

            2
        )

    else:

        placement_rate = 0


    # ========================================================
    # AVERAGE CGPA
    # ========================================================

    if "CGPA" in df.columns:

        average_cgpa = round(

            pd.to_numeric(

                df["CGPA"],

                errors="coerce"

            ).mean(),

            2
        )

    else:

        average_cgpa = 0


    # ========================================================
    # AVERAGE SALARY
    # ========================================================

    if (
        "Salary Package" in df.columns
        and
        "PlacementStatus" in df.columns
    ):

        placed_df = df[
            df["PlacementStatus"] == 1
        ]

        average_salary = round(

            pd.to_numeric(

                placed_df["Salary Package"],

                errors="coerce"

            ).mean(),

            2
        )

    else:

        average_salary = 0


    # ========================================================
    # EDA IMAGES
    # ========================================================

    eda_images = []

    if EDA_DESTINATION.exists():

        eda_images = sorted(

            [
                image.name

                for image
                in EDA_DESTINATION.glob("*.png")

            ]

        )


    # ========================================================
    # CLUSTERING IMAGES
    # ========================================================

    clustering_images = get_clustering_images()


    # ========================================================
    # PREPROCESSED DATA STATUS
    # ========================================================

    preprocessing_status = (
        check_preprocessed_data()
    )


    # ========================================================
    # PRINT DASHBOARD INFORMATION
    # ========================================================

    print("\nDashboard Statistics:")

    print(
        "Total Students:",
        total_students
    )

    print(
        "Total Features:",
        total_features
    )

    print(
        "Placed Students:",
        placed_students
    )

    print(
        "Not Placed Students:",
        not_placed_students
    )

    print(
        "Placement Rate:",
        placement_rate,
        "%"
    )

    print(
        "Average CGPA:",
        average_cgpa
    )

    print(
        "Average Salary:",
        average_salary
    )

    print(
        "Missing Values:",
        missing_values
    )

    print(
        "Duplicate Records:",
        duplicates
    )

    print(
        "EDA Images:",
        len(eda_images)
    )

    print(
        "Clustering Images:",
        len(clustering_images)
    )

    print(
        "Preprocessed Dataset:",
        preprocessing_status["available"]
    )

    print(
        "Preprocessed Features:",
        preprocessing_status["columns"]
    )


    # ========================================================
    # SEND DATA TO INDEX.HTML
    # ========================================================

    return render_template(

        "index.html",


        # ----------------------------------------------------
        # BASIC STATISTICS
        # ----------------------------------------------------

        total_students=total_students,

        total_features=total_features,


        # ----------------------------------------------------
        # PLACEMENT
        # ----------------------------------------------------

        placed_students=placed_students,

        not_placed_students=not_placed_students,

        placement_rate=placement_rate,


        # ----------------------------------------------------
        # DATA QUALITY
        # ----------------------------------------------------

        missing_values=missing_values,

        duplicates=duplicates,


        # ----------------------------------------------------
        # AVERAGES
        # ----------------------------------------------------

        average_cgpa=average_cgpa,

        average_salary=average_salary,


        # ----------------------------------------------------
        # DATASET INFORMATION
        # ----------------------------------------------------

        column_names=column_names,

        missing_report=missing_report,


        # ----------------------------------------------------
        # EDA
        # ----------------------------------------------------

        eda_images=eda_images,


        # ----------------------------------------------------
        # CLUSTERING
        # ----------------------------------------------------

        clustering_images=clustering_images,


        # ----------------------------------------------------
        # PREPROCESSING
        # ----------------------------------------------------

        preprocessing_available=(
            preprocessing_status["available"]
        ),

        preprocessed_features=(
            preprocessing_status["columns"]
        )

    )


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n============================================")
    print("Starting Placement Flask Dashboard")
    print("============================================")

    app.run(
        debug=True
    )