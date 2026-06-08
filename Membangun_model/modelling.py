import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# 1. Load data
X_train = pd.read_csv('Iris_preprocessing/X_train.csv')
y_train = pd.read_csv('Iris_preprocessing/y_train.csv')

# 2. Train model dengan parameter terbaik dari hasil eksperimen DagsHub
model = RandomForestClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train.values.ravel())

# 3. Simpan model ke file .pkl
joblib.dump(model, '../Monitoring dan Logging/model.pkl')