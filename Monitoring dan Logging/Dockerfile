FROM python:3.12-slim

WORKDIR /app

COPY Inference.py .
COPY model.pkl .
COPY prometheus_exporter.py .

RUN pip install flask pandas scikit-learn joblib prometheus-client

EXPOSE 5000

CMD ["python", "Inference.py"]