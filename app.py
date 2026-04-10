import streamlit as st
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

st.set_page_config(page_title="EcoData Dashboard", page_icon="🌱", layout="wide")

st.title("🌱 EcoData: Air Quality Dashboard")
st.markdown("Explore air quality data, discover clusters of pollution, and predict the Air Quality Index (AQI) using Machine Learning.")

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("data/city_day.csv")
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
    df_clean = df.copy()
    for col in ['City', 'AQI_Bucket', 'Date']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return df_clean, df

@st.cache_data
def run_clustering(df):
    features = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','AQI']
    features = [f for f in features if f in df.columns]
    X_cluster = df[features]
    
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(X_cluster)
    
    mbk = MiniBatchKMeans(n_clusters=3, batch_size=1000, random_state=42)
    clusters = mbk.fit_predict(X_cluster_scaled)
    df_clustered = df.copy()
    df_clustered['Cluster'] = clusters
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_cluster_scaled)
    df_clustered['PCA1'] = X_pca[:,0]
    df_clustered['PCA2'] = X_pca[:,1]
    
    return df_clustered

@st.cache_resource
def train_model(df):
    X = df.drop(columns=['AQI'])
    # In case there are PCA cols or clusters, drop them specifically if they exist in df
    if 'PCA1' in X.columns:
        X = X.drop(columns=['PCA1', 'PCA2', 'Cluster'], errors='ignore')
    y = df['AQI']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    
    y_pred = rf_model.predict(X_test_scaled)
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }
    return rf_model, metrics, y_test, y_pred, scaler

df_clean, df_preprocessed = load_and_preprocess_data()

tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🧩 Clustering Analysis", "🤖 AQI Prediction"])

with tab1:
    st.header("Dataset Overview")
    st.write(f"The dataset contains {df_clean.shape[0]} rows and {df_clean.shape[1]} features after cleanup.")
    st.dataframe(df_clean.head(500))
    
    st.subheader("AQI Distribution")
    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
    sns.histplot(df_clean['AQI'], bins=50, kde=True, ax=ax, color='teal')
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    st.header("K-Means Clustering Analysis")
    df_clustered = run_clustering(df_preprocessed)
    st.write("We used MiniBatchKMeans to group data into 3 clusters based on air pollutant features.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cluster Distribution")
        fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=100)
        sns.countplot(x='Cluster', data=df_clustered, palette='Set2', ax=ax2)
        st.pyplot(fig2)
        plt.close(fig2)
        
    with col2:
        st.subheader("PCA Visualization")
        plot_data = df_clustered.sample(n=min(len(df_clustered), 5000), random_state=42)
        if len(df_clustered) > len(plot_data):
            st.caption(f"Showing a random sample of {len(plot_data)} points for the PCA scatter plot to preserve performance.")
        fig3, ax3 = plt.subplots(figsize=(8, 6), dpi=100)
        sns.scatterplot(
            x='PCA1', y='PCA2', hue='Cluster', palette='Set2',
            data=plot_data, s=40, alpha=0.6, ax=ax3
        )
        st.pyplot(fig3)
        plt.close(fig3)

with tab3:
    st.header("Random Forest Regressor")
    with st.spinner('Training model... (This might take a minute)'):
        model, metrics, y_test, y_pred, scaler = train_model(df_preprocessed)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R-Squared Score", f"{metrics['R2']:.4f}")
    col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
    col3.metric("MAE", f"{metrics['MAE']:.2f}")
    col4.metric("MSE", f"{metrics['MSE']:.2f}")
    
    st.subheader("Actual vs Predicted AQI")
    fig4, ax4 = plt.subplots(figsize=(10, 5), dpi=100)
    ax4.scatter(y_test, y_pred, alpha=0.3, color='blue')
    ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    ax4.set_xlabel("Actual AQI")
    ax4.set_ylabel("Predicted AQI")
    st.pyplot(fig4)
    plt.close(fig4)
