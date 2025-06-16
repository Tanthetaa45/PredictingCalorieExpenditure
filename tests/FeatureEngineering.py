from sklearn.preprocessing import LabelEncoder
import numpy as np


label_enc = LabelEncoder()
label_enc.fit(train_df['Sex'])


train_df['Sex'] = label_enc.transform(train_df['Sex'])
test_df['Sex'] = label_enc.transform(test_df['Sex'])


def feature_engineering(data, numeric_cols):
    data['BMI'] = data['Weight'] / (data['Height'] / 100)**2
    data['Intensity'] = data['Heart_Rate'] / data['Duration']
    for i in range(len(numeric_cols)):
        f1 = numeric_cols[i]
        for j in range(i+1, len(numeric_cols)):
            f2 = numeric_cols[j]
            data[f'{f1}_x_{f2}'] = data[f1] * data[f2]
    return data


numeric_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']


train_df = feature_engineering(train_df, numeric_cols)
test_df = feature_engineering(test_df, numeric_cols)


X = train_df.drop(['Calories'], axis=1)
y = np.log1p(train_df['Calories'].values)

#PCA

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
                               