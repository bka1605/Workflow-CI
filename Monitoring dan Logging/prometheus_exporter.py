from prometheus_client import Counter, Summary

REQUEST_COUNT = Counter('app_request_count', 'Total request yang masuk ke aplikasi', ['method', 'endpoint', 'http_status'])
PREDICTION_COUNT = Counter('model_prediction_count','Total hasil prediksi model',['prediction_result'])