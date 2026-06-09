import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import joblib

mlflow.autolog()

def run_training():    
    X_train = pd.read_csv('mobile_price_cls_preprocessing/X_train.csv')
    y_train = pd.read_csv('mobile_price_cls_preprocessing/y_train.csv')
    X_test = pd.read_csv('mobile_price_cls_preprocessing/X_test.csv')
    y_test = pd.read_csv('mobile_price_cls_preprocessing/y_test.csv')

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, max_depth=5)
        model.fit(X_train, y_train.values.ravel())
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        print(f"Training Accuracy: {train_acc}")
        print(f"Testing Accuracy: {test_acc}")
        
if __name__ == "__main__":
    run_training()