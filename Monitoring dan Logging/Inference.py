from flask import Flask, request, jsonify, Response
import joblib
import pandas as pd
from prometheus_client import Counter, generate_latest

app = Flask(__name__)

# load model
model = joblib.load('model.pkl')

# siapin Prometheus
REQUEST_COUNT = Counter(
    'app_request_count', 
    'Total request yang masuk ke aplikasi', 
    ['method', 'endpoint', 'http_status']
)
PREDICTION_COUNT = Counter(
    'model_prediction_count',
    'Total hasil prediksi model',
    ['prediction_result']
)

# endpoint Prometheus
@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype="text/plain")

# endpoint Prediksi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ambil json user
        data = request.json
        
        df = pd.DataFrame([data])
        
        # predict
        prediction = model.predict(df)
        result = int(prediction[0])
        
        REQUEST_COUNT.labels('POST', '/predict', 200).inc()
        PREDICTION_COUNT.labels(prediction_result=str(result)).inc()
        
        return jsonify({'status': 'success', 'prediction': result})
        
    except Exception as e:
        REQUEST_COUNT.labels('POST', '/predict', 500).inc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("API berjalan di http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000)