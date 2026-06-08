import requests

url = 'http://127.0.0.1:5000/predict'

# dummy test
data_input = {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2
}

print("Mengirim dummy utk predict")
response = requests.post(url, json=data_input)

print("Status Code:", response.status_code)
print("Hasil Prediksi:", response.json())