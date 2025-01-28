import requests
import json
import pandas as pd

features = [
    'ProviderId', 'ProductCategory', 'Amount', 'Value', 'PricingStrategy', 'FraudResult',
    'Total_Transaction_Amount', 'Average_Transaction_Amount', 'Transaction_Count', 'Std_Transaction_Amount',
    'Transaction_Hour', 'Transaction_Day', 'Transaction_Month', 'Transaction_Year', 'CurrencyCode_WOE',
    'ProviderId_WOE', 'ProductId_WOE', 'ProductCategory_WOE', 'Recency', 'RFMS_score', 'ProductId_1',
    'ProductId_2', 'ProductId_3', 'ProductId_4', 'ProductId_5', 'ProductId_6', 'ProductId_7', 'ProductId_8',
    'ProductId_9', 'ProductId_10', 'ProductId_11', 'ProductId_12', 'ProductId_13', 'ProductId_14',
    'ProductId_15', 'ProductId_16', 'ProductId_17', 'ProductId_18', 'ProductId_19', 'ProductId_20',
    'ProductId_21', 'ProductId_22', 'ChannelId_ChannelId_2', 'ChannelId_ChannelId_3', 'ChannelId_ChannelId_5',
    'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionWeekday', 'CurrencyCode', 'CountryCode', 'ProductId'
]




input_values = [
    5, 0, -0.046371, -0.072291, 2, 0, 0.170118, -0.067623, -0.311831, 0.165122,
    2, 15, 11, 2018, 0.0, 3.137005, 1.645067, 1.620379, 2265, -0.042337, False,
    False, False, False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False, True, False,
    False, False, 2, 15, 11, 3, 0, 1, 1
]






# Ensure the input_values list length matches the number of features
print("Number of features:", len(features))
print("Number of input values:", len(input_values))

# Create a DataFrame
df = pd.DataFrame([input_values], columns=features)

# Convert DataFrame to dictionary
payload = df.to_dict(orient='records')[0]

# Send POST request to the API
url = 'http://127.0.0.1:8000/api/predict/'
headers = {'Content-Type': 'application/json'}
response = requests.post(url, headers=headers, data=json.dumps(payload))

print("Status Code:", response.status_code)
print("Response Text:", response.text)
try:
    print("Response JSON:", response.json())
except json.JSONDecodeError:
    print("Response content is not valid JSON.")
