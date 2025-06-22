pca=PCA(n_components=3)
X_pca=pca.fit_transform(X.drop('Sex',axis=1))
kmeans_per_k=[KMeans(n_clusters=k,random_state=42).fit(X_pca) for k in range(1,10)]
inertias=[model.inertia_ for model in kmeans_per_k]
plt.plot(range(1,10),inertias,"bo-")
plt.xlabel("$k$")
plt.ylabel("Inertia")
plt.grid(True)
plt.title("Elbow Method for Optimal K")
plt.show()
kmeans=KMeans(n_clusters=3,random_state=42).fit(X_pca)
pca_df=pd.DataFrame(X_pca,columns=['PCA1','PCA2','PCA3'])
pca_df['cluster']=kmeans.labels_
pca_df['target_group']=pd.qcut(train_df['Calories'],q=3,labels=False)
accuracy=(pca_df['cluster']==pca_df['target_group']).mean()
print(f"Accuracy between clusters and target groups:{accuracy:.2%}")
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(pca_df['PCA1'], pca_df['PCA2'], pca_df['PCA3'], c=pca_df['target_group'], cmap='viridis', alpha=0.7)
plt.title("3D PCA Projection with Target Group Coloring")
plt.colorbar(scatter, label='Target Group')
plt.show()


                               import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

# Step 1: Drop non-numeric column and scale
X_features = X.drop('Sex', axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# Step 2: Apply PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Step 3: Elbow method to find optimal k
kmeans_per_k = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_pca) for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]

plt.plot(range(1, 10), inertias, "bo-")
plt.xlabel("$k$")
plt.ylabel("Inertia")
plt.grid(True)
plt.title("Elbow Method for Optimal K")
plt.show()

# Step 4: Fit final KMeans with k=3 (or based on elbow)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_pca)

# Step 5: Create DataFrame with PCA and clustering info
pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2', 'PCA3'])
pca_df['cluster'] = kmeans.labels_
pca_df['target_group'] = pd.qcut(train_df['Calories'], q=3, labels=False)

# Step 6: Evaluate clustering performance
ari = adjusted_rand_score(pca_df['target_group'], pca_df['cluster'])
nmi = normalized_mutual_info_score(pca_df['target_group'], pca_df['cluster'])

print(f"Adjusted Rand Index (ARI): {ari:.2%}")
print(f"Normalized Mutual Information (NMI): {nmi:.2%}")

# Step 7: 3D PCA plot colored by target group
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pca_df['PCA1'], pca_df['PCA2'], pca_df['PCA3'],
                     c=pca_df['target_group'], cmap='viridis', alpha=0.7)
ax.set_title("3D PCA Projection with Target Group Coloring")
plt.colorbar(scatter, label='Target Group')
plt.show()

# Step 8: 3D PCA plot colored by cluster
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pca_df['PCA1'], pca_df['PCA2'], pca_df['PCA3'],
                     c=pca_df['cluster'], cmap='tab10', alpha=0.7)
ax.set_title("3D PCA Projection with KMeans Cluster Coloring")
plt.colorbar(scatter, label='Cluster')
plt.show()