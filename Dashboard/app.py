from flask import Flask, render_template
import pandas as pd
from pathlib import Path
import shutil

app = Flask(__name__)

# =================================================
# PATHS
# =================================================

BASE_DIR = Path(__file__).resolve().parent

# placementprediction.csv is outside Dashboard folder
DATA_FILE = BASE_DIR.parent / "placementprediction.csv"

# EDA.py generates graphs here
EDA_SOURCE = BASE_DIR.parent / "eda_output"

# Flask dashboard reads graphs from here
EDA_DESTINATION = BASE_DIR / "static" / "eda_images"


# =================================================
# COPY EDA IMAGES
# =================================================

def copy_eda_images():

    EDA_DESTINATION.mkdir(parents=True, exist_ok=True)

    if EDA_SOURCE.exists():

        for image in EDA_SOURCE.glob("*.png"):

            destination = EDA_DESTINATION / image.name

            shutil.copy2(image, destination)


# =================================================
# LOAD DATA
# =================================================

def load_data():

    return pd.read_csv(DATA_FILE)


# =================================================
# DASHBOARD
# =================================================

@app.route("/")
def home():

    # Copy EDA graphs automatically
    copy_eda_images()

    # Load dataset
    df = load_data()

    # ---------------------------------------------
    # BASIC STATISTICS
    # ---------------------------------------------

    total_students = len(df)

    total_features = len(df.columns)

    placed_students = int(
        (df["PlacementStatus"] == 1).sum()
    )

    not_placed_students = int(
        (df["PlacementStatus"] == 0).sum()
    )

    missing_values = int(
        df.isnull().sum().sum()
    )

    duplicates = int(
        df.duplicated().sum()
    )

    placement_rate = round(
        (placed_students / total_students) * 100,
        2
    )

    average_cgpa = round(
        df["CGPA"].mean(),
        2
    )

    # ---------------------------------------------
    # SALARY ANALYSIS
    # ---------------------------------------------

    placed_df = df[
        df["PlacementStatus"] == 1
    ]

    average_salary = round(
        placed_df["Salary Package"].mean(),
        2
    )

    # ---------------------------------------------
    # GET EDA GRAPH NAMES
    # ---------------------------------------------

    eda_images = []

    if EDA_DESTINATION.exists():

        eda_images = sorted(
            [
                image.name
                for image in EDA_DESTINATION.glob("*.png")
            ]
        )

    # ---------------------------------------------
    # SEND DATA TO INDEX.HTML
    # ---------------------------------------------

    return render_template(

        "index.html",

        total_students=total_students,
        total_features=total_features,

        placed_students=placed_students,
        not_placed_students=not_placed_students,

        missing_values=missing_values,
        duplicates=duplicates,

        placement_rate=placement_rate,
        average_cgpa=average_cgpa,
        average_salary=average_salary,

        eda_images=eda_images
    )


# =================================================
# RUN APPLICATION
# =================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )