import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

# 1. Koneksi ke DagsHub
mlflow.set_tracking_uri("https://dagshub.com/bka1605/Modelling_SML_Brian-Kristanto-A.mlflow")

# 2. Load data hasil preprocessing Kriteria 1
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')

# 3. Setup MLflow Experiment
mlflow.set_experiment("Iris-Classification-Tuning")

# 4. Training dengan MLflow
with mlflow.start_run():
    # Hyperparameter
    n_estimators = 100
    max_depth = 5
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train.values.ravel())
    
    # Log ke DagsHub
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.sklearn.log_model(model, "model")
    
    print("Model berhasil dilatih dan dikirim ke DagsHub!")
    
    # Tambahkan ini di dalam blok 'with mlflow.start_run():'
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Log metrik ke DagsHub
    mlflow.log_metric("accuracy", accuracy)
    print(f"Accuracy: {accuracy}")