# ============================================================
# K-MEANS OUTLIER DETECTION
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


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


# ============================================================
# SELECT FEATURES
# ============================================================

FEATURES = [
    "numerical__CGPA",
    "numerical__AptitudeTestScore"
]


for feature in FEATURES:

    if feature not in data.columns:

        print("\nERROR: Feature not found:")
        print(feature)

        raise SystemExit


X = data[FEATURES]


# ============================================================
# K-MEANS
# ============================================================

K = 3


print("\nApplying K-Means with K =", K)


kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)


clusters = kmeans.fit_predict(X)


# ============================================================
# CALCULATE DISTANCE FROM CENTROID
# ============================================================

centers = kmeans.cluster_centers_

distances = []


for index, row in X.iterrows():

    cluster_number = clusters[index]

    center = centers[cluster_number]

    distance = (
        (row.iloc[0] - center[0]) ** 2
        +
        (row.iloc[1] - center[1]) ** 2
    ) ** 0.5

    distances.append(distance)


data["Cluster"] = clusters

data["Distance_From_Center"] = distances


# ============================================================
# OUTLIER THRESHOLD
# ============================================================

mean_distance = data[
    "Distance_From_Center"
].mean()

std_distance = data[
    "Distance_From_Center"
].std()


threshold = (
    mean_distance
    +
    2 * std_distance
)


print("\nMean Distance:")
print(f"{mean_distance:.4f}")


print("\nStandard Deviation:")
print(f"{std_distance:.4f}")


print("\nOutlier Threshold:")
print(f"{threshold:.4f}")


# ============================================================
# IDENTIFY OUTLIERS
# ============================================================

data["KMeans_Outlier"] = (
    data["Distance_From_Center"]
    > threshold
).astype(int)


# ============================================================
# DISPLAY RESULTS
# ============================================================

outlier_count = data[
    "KMeans_Outlier"
].sum()


normal_count = len(data) - outlier_count


print("\n============================================")
print("K-MEANS OUTLIER RESULTS")
print("============================================")


print(
    "\nNormal Data Points:",
    normal_count
)


print(
    "Outliers:",
    outlier_count
)


print("\nSample Results:")

print(
    data[
        FEATURES
        + [
            "Cluster",
            "Distance_From_Center",
            "KMeans_Outlier"
        ]
    ].head(10)
)


# ============================================================
# SAVE OUTPUT
# ============================================================

OUTPUT_FILE = r"C:\Users\ASUS\Downloads\ML PROGRAMS - Copy\kmeans_outliers_output.csv"


data.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nOutput File:")

print(OUTPUT_FILE)


# ============================================================
# VISUALIZATION
# ============================================================

normal_data = data[
    data["KMeans_Outlier"] == 0
]

outlier_data = data[
    data["KMeans_Outlier"] == 1
]


plt.figure(figsize=(8, 6))


# Normal points
plt.scatter(
    normal_data["numerical__CGPA"],
    normal_data["numerical__AptitudeTestScore"],
    alpha=0.4,
    label="Normal"
)


# Outlier points
plt.scatter(
    outlier_data["numerical__CGPA"],
    outlier_data["numerical__AptitudeTestScore"],
    marker="x",
    s=100,
    label="Outlier"
)


# Cluster centers
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=250,
    label="Cluster Centers"
)


plt.xlabel("CGPA")

plt.ylabel("Aptitude Test Score")

plt.title(
    "K-Means Based Outlier Detection"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n============================================")
print("K-MEANS OUTLIER DETECTION COMPLETED")
print("============================================")