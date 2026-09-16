import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

data = pd.read_csv(
    r"C:\Users\ASUS\Downloads\ML PROGRAMS - Copy\placement_preprocessed.csv"
)

X = data.drop(labels="PlacementStatus", axis=1)
X = X.sample(n=100, random_state=42)

methods = ['single', 'complete', 'average', 'ward']

for method in methods:

    Z = linkage(X, method=method, metric='euclidean')

    # ADDED: Display height/distance at which merges happen
    print("\n", method.upper(), "LINKAGE")
    print("Merge heights:")
    print(Z[:, 2])

    plt.figure(figsize=(8, 5))

    dendrogram(Z)

    plt.title(method.capitalize() + " Linkage")
    plt.xlabel("Students")
    plt.ylabel("Distance / Merge Height")

    plt.show()