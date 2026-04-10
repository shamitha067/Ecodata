

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1️⃣ Load dataset
df = pd.read_csv("data/city_day.csv")

print("Dataset Loaded Successfully\n")

# 2️⃣ Convert Date column
df['Date'] = pd.to_datetime(df['Date'])

# 3️⃣ Drop rows where AQI is missing (very important)
df = df.dropna(subset=['AQI'])

# 4️⃣ Drop columns with too many missing values
df.drop(columns=['Xylene'], inplace=True)

# 5️⃣ Fill remaining numeric missing values with median
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# 6️⃣ Remove duplicates
df.drop_duplicates(inplace=True)

# 7️⃣ Create time-based features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# 8️⃣ Drop unnecessary columns
df.drop(columns=['City', 'AQI_Bucket', 'Date'], inplace=True)

print("\nFinal Dataset Shape:", df.shape)

# 9️⃣ Separate Features & Target
X = df.drop('AQI', axis=1)
y = df['AQI']

# 🔟 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1️⃣1️⃣ Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nPreprocessing Completed Successfully ")
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)