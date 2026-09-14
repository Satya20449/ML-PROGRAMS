# ============================================================
# K-MEANS FINAL
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


print("\nSelected Features:")
print(X.head())


# ============================================================
# SELECT K
# ============================================================

K = 3


print("\nNumber of Clusters:", K)


# ============================================================
# CREATE K-MEANS MODEL
# ============================================================

kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)


# ============================================================
# FIT MODEL
# ============================================================

print("\nTraining K-Means model...")

clusters = kmeans.fit_predict(X)


# ============================================================
# ADD CLUSTER
# ============================================================

data["Cluster"] = clusters


# ============================================================
# CLUSTER CENTERS
# ============================================================

print("\nCluster Centers:")

centers = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=FEATURES
)

print(centers)


# ============================================================
# CLUSTER COUNTS
# ============================================================

print("\nNumber of Students in Each Cluster:")

print(
    data["Cluster"]
    .value_counts()
    .sort_index()
)


# ============================================================
# SAMPLE RESULTS
# ============================================================

print("\nSample Cluster Results:")

print(
    data[
        FEATURES + ["Cluster"]
    ].head(10)
)


# ============================================================
# SAVE CLUSTERED DATA
# ============================================================

OUTPUT_FILE = r"C:\Users\ASUS\Downloads\ML PROGRAMS - Copy\kmeans_final_output.csv"

data.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nClustered dataset saved to:")

print(OUTPUT_FILE)


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


# Cluster centers
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=250,
    label="Cluster Centers"
)


plt.xlabel("CGPA")

plt.ylabel("Aptitude Test Score")

plt.title(
    "Final K-Means Clustering"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n============================================")
print("FINAL K-MEANS COMPLETED")
print("============================================")