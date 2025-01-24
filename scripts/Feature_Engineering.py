import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class Feature_Engineering:
    
    def __init__(self, file_path):
        self.data = self.load_data(file_path)
    
    def load_data(self, file_path):
        """Load the dataset."""
        data = pd.read_csv(file_path)
        return data
    def total_transaction_amount(self):
        """Calculate Total Transaction Amount for each customer."""
        total_transaction_amount = self.data.groupby('CustomerId')['Amount'].sum().reset_index()
        total_transaction_amount.columns = ['CustomerId', 'total_transaction_amount']
        return total_transaction_amount
    def average_transaction_amount(self):
        """Calculate Average Transaction Amount for each customer."""
        average_transaction_amount = self.data.groupby('CustomerId')['Amount'].mean().reset_index()
        average_transaction_amount.columns = ['CustomerId', 'average_transaction_amount']
        return average_transaction_amount