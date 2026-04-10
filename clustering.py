
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

chunksize = 5000  
chunks = []

print("Loading dataset in chunks...")
for chunk in pd.read_csv('cleaned_city_day.csv', chunksize=chunksize):
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
print("Dataset loaded:", df.shape)


features = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene','AQI']
X = df[features]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data scaling completed.")

inertia = []
K_range = range(1, 11)

for k in K_range:
    km = MiniBatchKMeans(n_clusters=k, batch_size=1000, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()



k = 3  # change this after checking Elbow plot
mbk = MiniBatchKMeans(n_clusters=k, batch_size=1000, random_state=42)
clusters = mbk.fit_predict(X_scaled)

df['Cluster'] = clusters
print("Clustering completed. Sample of cluster assignments:")
print(df[['Cluster'] + features].head())



pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:,0]
df['PCA2'] = X_pca[:,1]

plt.figure(figsize=(8,6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='Set2', data=df, s=50)
plt.title('Clusters Visualization (PCA Reduced)')
plt.show()


df.to_csv('clustered_city_day.csv', index=False)
print("Clustered dataset saved as clustered_city_day.csv")