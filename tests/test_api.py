import requests
import json

url = 'http://127.0.0.1:8000/api/predict/'
headers = {'Content-Type': 'application/json'}
data = {
    "Std_Transaction_Amount": 200.0,
    "CurrencyCode": 0,
    "CountryCode": 256,
    "ProviderId": 5,
    "ProductId": 1,
    "TransactionHour": 14,
    "TransactionDay": 15,
    "TransactionMonth": 11,
    "TransactionWeekday": 2,
    "CurrencyCode_WOE": 0.0,
    "ProviderId_WOE": 3.137005,
    "ProductId_WOE": 1.645067,
    "ProductCategory_WOE": 1.620379,
    "Recency": 2265,
    "RFMS_score": -0.042337
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print("Status Code:", response.status_code)
print("Response Text:", response.text)
try:
    print("Response JSON:", response.json())
except json.JSONDecodeError:
    print("Response content is not valid JSON.")
