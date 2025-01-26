import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Mall_Customers.csv')
print("First few rows of the dataset:")
print(df.head())

# Data Cleaning: Rename columns for easier use
df.rename(columns={
    'Annual Income (k$)': 'AnnualIncome',
    'Spending Score (1-100)': 'SpendingScore'
}, inplace=True)
print("\nDataset after renaming columns:")
print(df.head())

# Feature Engineering: Add a 'Recency' column (all customers are considered active)
df['Recency'] = 0

# Normalize the data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(df[['Recency', 'AnnualIncome', 'SpendingScore']])

# Elbow Method to find optimal clusters
wcss = []  # Within-Cluster-Sum-of-Squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.savefig('elbow_plot.png')  # Save Elbow Method plot
print("\nElbow Method plot saved as 'elbow_plot.png'")
plt.show()

# Perform clustering (assuming 5 clusters based on the elbow method)
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(rfm_scaled)
print("\nDataset with cluster labels:")
print(df.head())

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='AnnualIncome', y='SpendingScore', hue='Cluster', data=df, palette='viridis')
plt.title('Customer Segments')
plt.savefig('scatter_plot.png')  # Save scatter plot
print("Scatter plot saved as 'scatter_plot.png'")
plt.show()

# Cluster summary
cluster_summary = df.groupby('Cluster').agg({
    'Age': 'mean',
    'AnnualIncome': 'mean',
    'SpendingScore': 'mean'
}).reset_index()
print("\nCluster summary:")
print(cluster_summary)