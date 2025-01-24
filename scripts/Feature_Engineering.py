# Assuming the following imports and class definition are already done
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


