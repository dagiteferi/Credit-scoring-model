import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import logging
import numpy as np

class WOE:
    def __init__(self):
        self.iv_values_ = {}

    def fit(self, X, y):
        self.woe_dict = {}
        self.iv_values_ = {}
        for column in X.columns:
            self.woe_dict[column] = {}
            total_good = (y == 0).sum()
            total_bad = (y == 1).sum()
            for value in X[column].unique():
                good = ((X[column] == value) & (y == 0)).sum()
                bad = ((X[column] == value) & (y == 1)).sum()
                woe = np.log((good / total_good) / (bad / total_bad))
                iv = ((good / total_good) - (bad / total_bad)) * woe
                self.woe_dict[column][value] = woe
                self.iv_values_.setdefault(column, 0)
                self.iv_values_[column] += iv

    def transform(self, X):
        X_woe = X.copy()
        for column in X.columns:
            X_woe[column] = X_woe[column].map(self.woe_dict[column])
        return X_woe

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Feature_Engineering:
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

def encoding(data, target_variable='FraudResult', sample_size=10000):
    logger.info("encoding the categorical variables")
    try:
        # Apply Label Encoding for ordinal categorical variables first
        logger.info("label encoding for ordinal categorical variables")
        # (Your existing label encoding code)

        # Check data types of columns for WOE encoding
        logger.info("Checking data types of columns for WOE encoding")
        logger.info(data.dtypes)

        # Inspect unique values before conversion
        for col in ['CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory']:
            logger.info(f"Unique values in {col}: {data[col].unique()}")

        # Convert categorical to numeric using category codes
        for col in ['CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory']:
            if data[col].dtype == 'object':
                data[col] = data[col].astype('category').cat.codes

        logger.info("Data types after conversion:")
        logger.info(data.dtypes)

        # Now apply WOE encoding to a sample of the data
        logger.info("applying WOE encoding to a sample of the data...")
        woe = WOE()
        try:
            sample_data = data.sample(n=sample_size, random_state=1)
            woe.fit(sample_data[['CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory']], sample_data[target_variable])
            data_woe = woe.transform(data[['CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory']])
            data = pd.concat([data, data_woe], axis=1)
            iv_values = woe.iv_values_
            logger.info(f"Information Value (IV) for features: {iv_values}")
        except Exception as e:
            logger.error(f"Error during WOE transformation: {e}")

        # Now apply one-hot encoding for nominal variables
        logger.info("one-hot encoding for nominal variables")
        # (Your existing one-hot encoding code)

        return data

    except Exception as e:
        logger.error(f"error occurred {e}")
        return data  

def check_missing_values(df):
        """Check for missing values in each column of the DataFrame."""
        missing_values = df.isnull().sum()
        print("Missing values in each column:\n", missing_values)
        return missing_values
def fill_missing_values_with_mode(df, column_name):
        """Fill missing values in the specified column with the mode."""
        df[column_name].fillna(df[column_name].mode()[0], inplace=True)
        print(f"Missing values in '{column_name}' after imputation:\n", df[column_name].isnull().sum())

