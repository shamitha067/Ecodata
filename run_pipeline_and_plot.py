import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Imports loaded successfully!")

# Cell 2: Preprocessing
df = pd.read_csv("data/city_day.csv")
print("Dataset Loaded Successfully. Original Shape:", df.shape)

df['Date'] = pd.to_datetime(df['Date'])
df = df.dropna(subset=['AQI'])

if 'Xylene' in df.columns:
    df.drop(columns=['Xylene'], inplace=True)

numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

df.drop_duplicates(inplace=True)

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

for col in ['City', 'AQI_Bucket', 'Date']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

print("Final Dataset Shape after preprocessing:", df.shape)

# Cell 3: Clustering
features = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','AQI']
features = [f for f in features if f in df.columns]
X_cluster = df[features]

scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)
print("Data scaling completed for clustering.")

k = 3 
mbk = MiniBatchKMeans(n_clusters=k, batch_size=1000, random_state=42)
clusters = mbk.fit_predict(X_cluster_scaled)

df['Cluster'] = clusters
print("Clustering completed. Sample of cluster assignments:")
print(df[['Cluster'] + features].head())

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster_scaled)
df['PCA1'] = X_pca[:,0]
df['PCA2'] = X_pca[:,1]

plt.figure(figsize=(8,6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='Set2', data=df, s=50)
plt.title('Clusters Visualization (PCA Reduced)')
plt.savefig('cluster_visualization.png', bbox_inches='tight')
print("Saved cluster visualization to cluster_visualization.png")
plt.close()

# Cell 4: Model Training
X = df.drop(columns=['AQI', 'PCA1', 'PCA2'])
y = df['AQI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_model = StandardScaler()
X_train_scaled = scaler_model.fit_transform(X_train)
X_test_scaled = scaler_model.transform(X_test)

print("Training Data Shape:", X_train_scaled.shape)
print("Testing Data Shape:", X_test_scaled.shape)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
print("Training Random Forest Regressor...")
rf_model.fit(X_train_scaled, y_train)
print("Model Training Completed!")

# Cell 5: Evaluation
y_pred = rf_model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared Score: {r2:.4f}")

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI Values")
plt.savefig('actual_vs_predicted.png', bbox_inches='tight')
print("Saved actual vs predicted plot to actual_vs_predicted.png")
plt.close()
