import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import LabelEncoder , MinMaxScaler, StandardScaler
import datetime as dt

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
    
    def encoding(self, data, target_variable='FraudResult'):
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

            # Now apply WOE encoding
            logger.info("applying WOE encoding to certain categorical variables...")
            woe = WOE()
            try:
                woe.fit(data[['CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory']], data[target_variable])
                data_woe = woe.transform(data[['CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory']])
                data_woe.columns = [f"{col}_WOE" for col in data_woe.columns]  # Rename columns to avoid duplicates
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
            return data  # Return original data on error
def extract_features(data):
    logger.info("extracing some features")
    try:
        # Assuming the 'TransactionStartTime' is in string format, let's convert it to datetime
        data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])

        # Extracting features from 'TransactionStartTime'
        data['Transaction_Hour'] = data['TransactionStartTime'].dt.hour     # Hour of the transaction
        data['Transaction_Day'] = data['TransactionStartTime'].dt.day       # Day of the month
        data['Transaction_Month'] = data['TransactionStartTime'].dt.month   # Month of the year
        data['Transaction_Year'] = data['TransactionStartTime'].dt.year     # Year of the transaction
        return data
    except Exception as e:
        logger.error(f"error occured {e}")
def check_missing_values(df):
    """Check for missing values in each column of the DataFrame."""
    missing_values = df.isnull().sum()
    print("Missing values in each column:\n", missing_values)
    return missing_values

def fill_missing_values_with_mode(df, column_name):
    """Fill missing values in the specified column with the mode."""
    df[column_name].fillna(df[column_name].mode()[0], inplace=True)
    print(f"Missing values in '{column_name}' after imputation:\n", df[column_name].isnull().sum())

def Standardize_numeical_features(data):
    logger.info("normalize the numerical features")
    try:
        numerical_features = ['Amount', 'Value', 'Total_Transaction_Amount', 'Average_Transaction_Amount', 'Transaction_Count', 'Std_Transaction_Amount']
        # Normalize the Numerical Features (Range [0,1])
        # Initialize MinMaxScalar for normalization
        min_max_scalar =MinMaxScaler()

       
        standard_scalar = StandardScaler()
        # Apply standardization to the numerical columns
        data[numerical_features] = standard_scalar.fit_transform(data[numerical_features])

        # check the result 
        logger.info(f"the result of the standardized numeical featuresis  \n {data[numerical_features].head()}")
        return data
    except Exception as e:
        logger.error(f"error occured {e}")
def constructinf_RFMS_scores(data):
    logger.info("constructing the RFMS scores")
    try:
        logger.info("Calculate Recency as days since last transaction")
        
        # Convert TransactionStartTime to datetime
        data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])

        # Get the current date in UTC
        current_date = dt.datetime.now(dt.timezone.utc)

        # Calculate Recency as days since the last transaction
        data['Recency'] = (current_date - data['TransactionStartTime']).dt.days

        # Creating RFMS score; weight components based on their importance
        data['RFMS_score'] = (1 / (data['Recency'] + 1) * 0.4) + (data['Transaction_Count'] * 0.3) + (data['Total_Transaction_Amount'] * 0.3)
        
        logger.info("visualizing the RFMS space")
        visualuze_RFMS_space(data)
        
        logger.info("assigning the good and bad labels")
        assign_good_and_bad_lables(data)
        
        logger.info("calculating woe")
        # Ensure that the 'Label' and 'RFMS_score' columns exist in the DataFrame
        if 'Label' in data.columns and 'RFMS_score' in data.columns:
            woe_results = calculate_woe(data, 'Label', 'RFMS_score')
            logger.info("WoE results calculated successfully")
        else:
            logger.error("Columns 'Label' or 'RFMS_score' not found in DataFrame")
            return

        # Print WoE results
        print("WoE Results:")
        print(woe_results)
        
        return data
    
    except Exception as e:
        logger.error(f"error occurred {e}")

def visualuze_RFMS_space(data):
    try:
        # Scatter plot of RFMS scores
        plt.scatter(data['Transaction_Count'], data['Total_Transaction_Amount'], c=data['RFMS_score'], cmap='viridis')
        plt.colorbar(label='RFMS Score')
        plt.xlabel('Transaction Count')
        plt.ylabel('Total Transaction Amount')
        plt.title('RFMS Visualization')
        plt.show()
    except Exception as e:
        logger.error(f"Error in visualizing RFMS space: {e}")

def assign_good_and_bad_lables(data):
    threshold = data['RFMS_score'].median()
    data['Label'] = np.where(data['RFMS_score'] > threshold, 'Good', 'Bad')
    # Debugging prints
    print("RFMS Score and Labels after assignment:\n", data[['RFMS_score', 'Label']].head())
    print("Label Distribution after assignment:\n", data['Label'].value_counts())

def calculate_woe(df, target, feature):
    """
    Calculate the Weight of Evidence (WoE) for a given feature.
    """
    try:
        # Convert target to numeric if necessary
        df[target] = pd.to_numeric(df[target], errors='coerce')

        # Group by the feature and calculate the counts and events
        woe_df = df.groupby(feature).agg(
            count=(target, 'size'),
            event=(target, 'sum')
        ).reset_index()

        woe_df['non_event'] = woe_df['count'] - woe_df['event']
        
        # Debugging print statement
        print("WoE DataFrame:\n", woe_df)

        total_events = woe_df['event'].sum()
        total_non_events = woe_df['non_event'].sum()

        # Avoid division by zero and skip categories with zero events or non-events
        if total_events == 0 or total_non_events == 0:
            raise ValueError("Total events or non-events cannot be zero.")

        woe_df['event_rate'] = woe_df['event'] / total_events
        woe_df['non_event_rate'] = woe_df['non_event'] / total_non_events
        
        # Calculate WoE and handle zero events or non-events by skipping them
        woe_df['woe'] = np.where(
            (woe_df['event_rate'] == 0) | (woe_df['non_event_rate'] == 0),
            0,  # Assign a default value for WoE
            np.log(woe_df['event_rate'] / woe_df['non_event_rate']).replace([-np.inf, np.inf], 0)
        )

        return woe_df[[feature, 'count', 'event', 'non_event', 'woe']]
    
    except Exception as e:
        logger.error(f"Error in calculating WoE: {e}")
        return None
