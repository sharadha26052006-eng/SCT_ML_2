
# ==========================================================
# SkillCraft Technology - Machine Learning Internship
# Task 02: Customer Segmentation using K-Means Clustering
# ==========================================================

# 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 2. Load Dataset

print("\nLoading Customer Dataset...")

data = pd.read_csv("Mall_Customers.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", data.shape)

print("\nFirst 5 Customer Records:")
print(data.head())

# 3. Select Features

features = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# 4. Feature Scaling

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 5. Find Optimal Number of Clusters using Elbow Method

inertia = []

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot Elbow Graph

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')

plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.xticks(range(1, 11))
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("elbow_method.png", dpi=300)
plt.show()

# 6. Apply K-Means Clustering

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

data['Cluster'] = kmeans.fit_predict(scaled_features)

print("\nK-Means Clustering Completed!")

# 7. Evaluate Clustering

score = silhouette_score(scaled_features, data['Cluster'])

print("\n========== MODEL EVALUATION ==========")
print("Number of Clusters: 5")
print("Silhouette Score:", round(score, 4))

# 8. Display Customer Groups

print("\nCustomer Cluster Counts:")
print(data['Cluster'].value_counts().sort_index())

# 9. Save Clustered Results

data.to_csv("customer_segments.csv", index=False)

print("\nClustered Customer Data Saved Successfully!")

# 10. Visualize Customer Segments

plt.figure(figsize=(10, 6))

plt.scatter(
    data['Annual Income (k$)'],
    data['Spending Score (1-100)'],
    c=data['Cluster'],
    cmap='viridis',
    s=60
)

centers = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker='X',
    s=250,
    color='red',
    label='Cluster Centers'
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("customer_segmentation.png", dpi=300)
plt.show()

# 11. Display Cluster Characteristics

print("\n========== CUSTOMER GROUP ANALYSIS ==========")

cluster_summary = data.groupby('Cluster')[
    ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
].mean().round(2)

print(cluster_summary)

# 12. Final Status

print("\n==========================================")
print("TASK 02 COMPLETED SUCCESSFULLY!")
print("Customer Segmentation using K-Means")
print("==========================================")
