import requests
import json
import pandas as pd

# List of features used for training
features = ['Unnamed: 0', 'ProviderId', 'ProductCategory', 'Amount', 'Value', 'PricingStrategy', 
            'FraudResult', 'Total_Transaction_Amount', 'Average_Transaction_Amount', 
            'Transaction_Count', 'Std_Transaction_Amount', 'Transaction_Hour', 'Transaction_Day', 
            'Transaction_Month', 'Transaction_Year', 'CurrencyCode_WOE', 'ProviderId_WOE', 
            'ProductId_WOE', 'ProductCategory_WOE', 'Recency', 'RFMS_score', 'Label', 'ProductId_1', 
            'ProductId_2', 'ProductId_3', 'ProductId_4', 'ProductId_5', 'ProductId_6', 'ProductId_7', 
            'ProductId_8', 'ProductId_9', 'ProductId_10', 'ProductId_11', 'ProductId_12', 
            'ProductId_13', 'ProductId_14', 'ProductId_15', 'ProductId_16', 'ProductId_17', 
            'ProductId_18', 'ProductId_19', 'ProductId_20', 'ProductId_21', 'ProductId_22', 
            'ChannelId_ChannelId_2', 'ChannelId_ChannelId_3', 'ChannelId_ChannelId_5', 
            'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionWeekday']

# Example input values
input_values = [0, 5, 'Category_1', 200.0, 100.0, 'Strategy_1', 0, 1000.0, 100.0, 10, 200.0, 14, 
                15, 11, 2025, 0.0, 3.137005, 1.645067, 1.620379, 2265, -0.042337, 1, 0, 0, 0, 1, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 14, 15, 11, 2]

# Create a DataFrame
df = pd.DataFrame([input_values], columns=features)

# Remove features that are not needed or add necessary preprocessing steps
df = df.drop(columns=['Unnamed: 0', 'Label'])  # Drop unnecessary columns

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
