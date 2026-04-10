import json

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

cells = []

# Cell 1: Imports
cells.append(create_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Imports loaded successfully!")"""))

# Cell 2: Preprocessing
cells.append(create_code_cell("""# Load dataset
df = pd.read_csv("data/city_day.csv")
print("Dataset Loaded Successfully. Original Shape:", df.shape)

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])

# Drop rows where AQI is missing (Target variable)
df = df.dropna(subset=['AQI'])

# Drop columns with too many missing values
if 'Xylene' in df.columns:
    df.drop(columns=['Xylene'], inplace=True)

# Fill remaining numeric missing values with median
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Create time-based features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Drop unnecessary columns
for col in ['City', 'AQI_Bucket', 'Date']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

print("Final Dataset Shape after preprocessing:", df.shape)"""))

# Cell 3: Clustering
cells.append(create_code_cell("""# Define features for clustering
features = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','AQI']

# Using dropping method or ignoring if absent
features = [f for f in features if f in df.columns]
X_cluster = df[features]

# Scale
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)
print("Data scaling completed for clustering.")

# K-Means Clustering
k = 3 # Optimal K from Elbow method usually
mbk = MiniBatchKMeans(n_clusters=k, batch_size=1000, random_state=42)
clusters = mbk.fit_predict(X_cluster_scaled)

df['Cluster'] = clusters
print("Clustering completed. Sample of cluster assignments:")
print(df[['Cluster'] + features].head())

# PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster_scaled)
df['PCA1'] = X_pca[:,0]
df['PCA2'] = X_pca[:,1]

plt.figure(figsize=(8,6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='Set2', data=df, s=50)
plt.title('Clusters Visualization (PCA Reduced)')
plt.show()"""))

# Cell 4: Model Training
cells.append(create_code_cell("""# Prepare features and target
X = df.drop(columns=['AQI', 'PCA1', 'PCA2']) # Drop target and PCA cols used for viz only
y = df['AQI']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler_model = StandardScaler()
X_train_scaled = scaler_model.fit_transform(X_train)
X_test_scaled = scaler_model.transform(X_test)

print("Training Data Shape:", X_train_scaled.shape)
print("Testing Data Shape:", X_test_scaled.shape)

# Initialize and Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
print("Training Random Forest Regressor...")
rf_model.fit(X_train_scaled, y_train)
print("Model Training Completed!")"""))

# Cell 5: Evaluation
cells.append(create_code_cell("""# Predict on Test Set
y_pred = rf_model.predict(X_test_scaled)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared Score: {r2:.4f}")

# Plot Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI Values")
plt.show()"""))

notebook = {
    "cells": cells,
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("EcoData_Model.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("EcoData_Model.ipynb has been successfully created using json!")
