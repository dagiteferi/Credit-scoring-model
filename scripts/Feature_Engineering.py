# Assuming the following imports and class definition are already done
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class Feature_Engineering:
    # Existing class methods...
    def __init__(self, file_path):
        self.data = self.load_data(file_path)
    
    def load_data(self, file_path):
        """Load the dataset."""
        data = pd.read_csv(file_path)
        return data

    def total_transaction_amount(self):
        """Calculate Total Transaction Amount for each customer."""
        filtered_data = self.data[self.data['Amount'] > 0]  # Filter out negative values
        print("Filtered Data for Total Transaction Amount:\n", filtered_data.head())
        total_transaction_amount = filtered_data.groupby('CustomerId')['Amount'].sum().reset_index()
        total_transaction_amount.columns = ['CustomerId', 'total_transaction_amount']
        return total_transaction_amount
    
    def average_transaction_amount(self):
        """Calculate Average Transaction Amount for each customer."""
        filtered_data = self.data[self.data['Amount'] > 0]  # Filter out negative values
        print("Filtered Data for Average Transaction Amount:\n", filtered_data.head())
        average_transaction_amount = filtered_data.groupby('CustomerId')['Amount'].mean().reset_index()
        average_transaction_amount.columns = ['CustomerId', 'average_transaction_amount']
        return average_transaction_amount
    
# Define the standalone function outside the class
def creating_aggregate_features(data):
    try:
        aggregates = data.groupby('CustomerId').agg(
            Total_Transaction_Amount=('Amount', 'sum'),  # Sum of all transaction amounts per customer
            Average_Transaction_Amount=('Amount', 'mean'),  # Average transaction amount per customer
            Transaction_Count=('TransactionId', 'count'),  # Count of transactions per customer
            Std_Transaction_Amount=('Amount', 'std')  # Standard deviation of transaction amounts per customer
        ).reset_index()

        # Merge the aggregated features back into the original dataframe 'data'
        data = data.merge(aggregates, on='CustomerId', how='left')
        return data
    except Exception as e:
        print(f"Error occurred: {e}")

def extract_transaction_features(data):
    try:
        if not pd.api.types.is_datetime64_any_dtype(data['TransactionStartTime']):
            data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
        
        data['TransactionHour'] = data['TransactionStartTime'].dt.hour
        data['TransactionDay'] = data['TransactionStartTime'].dt.day
        data['TransactionMonth'] = data['TransactionStartTime'].dt.month
        data['TransactionYear'] = data['TransactionStartTime'].dt.year
        
        return data
    except Exception as e:
        print(f"Error occurred: {e}")



def encode_categorical_variables(ef, categorical_columns):
    try:
        # Apply One-Hot Encoding to most categorical columns
        onehot_encoder = OneHotEncoder(sparse=False, drop='first')  # 'drop=first' to avoid multicollinearity
        encoded_vars = onehot_encoder.fit_transform(ef[categorical_columns])
        encoded_vars_df = pd.DataFrame(encoded_vars, columns=onehot_encoder.get_feature_names_out(categorical_columns))
        ef = pd.concat([ef, encoded_vars_df], axis=1)
        
        # Apply Label Encoding to 'CurrencyCode' (if it's in the categorical columns)
        if 'CurrencyCode' in categorical_columns:
            label_encoder = LabelEncoder()
            ef['CurrencyCodeEncoded'] = label_encoder.fit_transform(ef['CurrencyCode'])
        
        return ef
    except Exception as e:
        print(f"Error occurred: {e}")

# Call the function with identified categorical columns





