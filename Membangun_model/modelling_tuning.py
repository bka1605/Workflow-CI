import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils import estimator_html_repr

# connect dagshub
mlflow.set_tracking_uri("https://dagshub.com/bka1605/Modelling_SML_Brian-Kristanto-A.mlflow")

# load data
X_train = pd.read_csv('Iris_preprocessing/X_train.csv')
y_train = pd.read_csv('Iris_preprocessing/y_train.csv')
X_test = pd.read_csv('Iris_preprocessing/X_test.csv')
y_test = pd.read_csv('Iris_preprocessing/y_test.csv')

# setup
mlflow.set_experiment("Iris-Classification-Tuning")

# train mlflow
with mlflow.start_run():
    n_estimators = 100
    max_depth = 5
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train.values.ravel())
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # log param & metric
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)
    
    # log model
    mlflow.sklearn.log_model(model, "model")
    
    # buat estimator.html
    with open("estimator.html", "w") as f:
        f.write(estimator_html_repr(model))
    mlflow.log_artifact("estimator.html")
    
    # metric info json
    with open("metric_info.json", "w") as f:
        json.dump({"accuracy": accuracy}, f)
    mlflow.log_artifact("metric_info.json")
    
    # gmbr train confus matrix
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    plt.title("Confusion Matrix")
    plt.savefig("training_confusion_matrix.png")
    mlflow.log_artifact("training_confusion_matrix.png")
    
    # classification report
    report = classification_report(y_test, predictions)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")
    
    # feature importance
    importances = model.feature_importances_
    fig2, ax2 = plt.subplots()
    ax2.barh(X_train.columns, importances)
    ax2.set_title("Feature Importances")
    plt.savefig("feature_importances.png")
    mlflow.log_artifact("feature_importances.png")

    print(f"Accuracy: {accuracy}")
    print("Model dan artefak selesai dikirim ke DagsHub")
    