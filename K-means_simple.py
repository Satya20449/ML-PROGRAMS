import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ============================================================
# LOAD PREPROCESSED DATASET
# ============================================================

data = pd.read_csv("placement_preprocessed.csv")

# ============================================================
# SELECT FEATURES
# ============================================================

X = data[
    [
        "numerical__CGPA",
        "numerical__AptitudeTestScore"
    ]
]

print("\nSelected Features:")
print(X.head())

# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(X.isnull().sum())

X = X.dropna()

print("\nFeatures after removing missing values:")
print(X.head())

# ============================================================
# K-MEANS CLUSTERING
# ============================================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X)

data.loc[X.index, "Cluster"] = clusters

# ============================================================
# CLUSTER RESULTS
# ============================================================

print("\nCluster Results:")

print(
    data.loc[
        X.index,
        [
            "numerical__CGPA",
            "numerical__AptitudeTestScore",
            "Cluster"
        ]
    ].head(10)
)

# ============================================================
# CLUSTER CENTERS
# ============================================================

print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# ============================================================
# NUMBER OF STUDENTS IN EACH CLUSTER
# ============================================================

print("\nNumber of Students in Each Cluster:")
print(
    data["Cluster"]
    .value_counts()
    .sort_index()
)

# ============================================================
# K-MEANS VISUALIZATION
# ============================================================

plt.scatter(
    X["numerical__CGPA"],
    X["numerical__AptitudeTestScore"],
    c=clusters
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=200
)

plt.xlabel("Standardized CGPA")
plt.ylabel("Standardized Aptitude Test Score")
plt.title("K-Means Clustering")

plt.show()

# ============================================================
# ELBOW METHOD - WCSS
# ============================================================

wcss = []

for k in range(1, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

# ============================================================
# WCSS GRAPH
# ============================================================

plt.plot(
    range(1, 11),
    wcss,
    marker="o"
)

plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal K")

plt.show()