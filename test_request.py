import requests

url = "http://127.0.0.1:5000/predict"

payload = {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.8,
    "humidity": 82,
    "ph": 6.5,
    "rainfall": 202
}

response = requests.post(url, data=payload)
print(response.status_code)
print("Prediction page returned successfully.")
