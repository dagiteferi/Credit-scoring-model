import requests
import json

url = 'http://127.0.0.1:8000/api/predict/'
headers = {'Content-Type': 'application/json'}
data = {
    "feature1": 1.0,
    "feature2": 2.0
    # Add other features as needed
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())
