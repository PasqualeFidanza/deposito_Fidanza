import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
df = pd.read_csv('Esercizio1\Mall_Customers.csv')

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
X_values = X.values

# Creazione del modello K-Means, fit e predict
km = KMeans(n_clusters=5, random_state=0)
labels = km.fit_predict(X_values)

plt.figure(figsize=(6, 6))
plt.scatter(X_values[:, 0], X_values[:, 1], c=labels, cmap='viridis')
plt.show()

centroids = km.cluster_centers_
print(centroids)

# Calcolo del cluster ad alto potenziale e plot del relativo centroide
scores = centroids[:, 0] + centroids[:, 1]
best_cluster = np.argmax(scores)
print(best_cluster)

best_customers = X_values[labels == best_cluster]

plt.figure(figsize=(6, 6))
plt.scatter(X_values[:, 0], X_values[:, 1], c=labels, cmap='viridis')
plt.scatter(
    centroids[best_cluster, 0],
    centroids[best_cluster, 1],
    c='red',
    s=250,
    marker='X',
    edgecolor='k',
    label='Best Cluster Centroid'
)
plt.show()

# Calcolo della distanza media
for l in np.unique(labels):
    cluster_points = X_values[labels == l]

    # Calcola la distanza euclidea di ogni punto dal centroide del cluster
    distances = np.linalg.norm(cluster_points - centroids[l], axis=1)

    # Calcola la distanza media
    mean_distance = np.mean(distances)

    print(f"Distanza media dei punti dal centroide {l}: {mean_distance:.2f}")