import requests

url = 'http://127.0.0.1:5000/predict'

# dummy test
data_input = {
    "battery_power": 1200,
    "blue": 1,
    "clock_speed": 2.2,
    "dual_sim": 0,
    "fc": 2,
    "four_g": 1,
    "int_memory": 16,
    "m_dep": 0.8,
    "mobile_wt": 130,
    "n_cores": 4,
    "pc": 5,
    "px_height": 800,
    "px_width": 1200,
    "ram": 2048,
    "sc_h": 12,
    "sc_w": 8,
    "talk_time": 15,
    "three_g": 1,
    "touch_screen": 1,
    "wifi": 1
}

print("Mengirim dummy utk predict")
for i in range(25):
    response = requests.post(url, json=data_input)
    print(f"Request ke-{i+1} - Status Code: {response.status_code})")
    print(f"Hasil Prediksi: {response.json()}")
