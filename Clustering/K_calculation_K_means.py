# ============================================================
# K CALCULATION USING ELBOW METHOD
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
# ELBOW METHOD
# ============================================================

print("\nCalculating WCSS...")

wcss = []

k_values = range(1, 11)


for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X)

    wcss.append(
        kmeans.inertia_
    )


# ============================================================
# DISPLAY WCSS
# ============================================================

print("\nWCSS Values:")

for k, value in zip(k_values, wcss):

    print(
        f"K = {k}  WCSS = {value:.4f}"
    )


# ============================================================
# ELBOW GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    list(k_values),
    wcss,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")

plt.ylabel("WCSS")

plt.title(
    "Elbow Method for Optimal K"
)

plt.xticks(
    list(k_values)
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n============================================")
print("ELBOW METHOD COMPLETED")
print("============================================")

print(
    "\nSelect the K value at the elbow point "
    "of the graph."
)

print("============================================")