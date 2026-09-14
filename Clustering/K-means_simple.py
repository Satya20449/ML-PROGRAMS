# ============================================================
# K-MEANS SIMPLE
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

print("\nFirst 5 Rows:")
print(data.head())


# ============================================================
# SELECT FEATURES
# ============================================================

FEATURES = [
    "numerical__CGPA",
    "numerical__AptitudeTestScore"
]


# Check features
for feature in FEATURES:

    if feature not in data.columns:

        print("\nERROR: Feature not found:")
        print(feature)

        raise SystemExit


X = data[FEATURES]


print("\nSelected Features:")
print(X.head())


# ============================================================
# K-MEANS MODEL
# ============================================================

print("\nApplying K-Means...")

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)


# ============================================================
# FIT MODEL
# ============================================================

clusters = kmeans.fit_predict(X)


# ============================================================
# ADD CLUSTER COLUMN
# ============================================================

data["Cluster"] = clusters


# ============================================================
# DISPLAY CLUSTERS
# ============================================================

print("\nCluster Centers:")

print(
    kmeans.cluster_centers_
)


print("\nCluster Counts:")

print(
    data["Cluster"].value_counts().sort_index()
)


print("\nSample Clustered Data:")

print(
    data[
        FEATURES + ["Cluster"]
    ].head(10)
)


# ============================================================
# VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    X["numerical__CGPA"],
    X["numerical__AptitudeTestScore"],
    c=clusters,
    alpha=0.5
)


# Plot cluster centers
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=200,
    label="Cluster Centers"
)


plt.xlabel("CGPA")

plt.ylabel("Aptitude Test Score")

plt.title("K-Means Clustering")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n============================================")
print("K-MEANS SIMPLE COMPLETED")
print("============================================")