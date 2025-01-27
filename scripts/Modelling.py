from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import logging

# Initialize logger
logger = logging.getLogger(__name__)

class Modeling:
    def __init__(self):
        pass

    def load_data(self, path):
        logger.info("Importing the data")
        try:
            # My data have 'Label' column for binary classification
            logger.info("Loading the data")
            data = pd.read_csv(path)
            return data
        except Exception as e:
            logger.error(f"Error occurred while loading the data: {e}")
            return None

    def preprocess_data(self, data):
        """
        Preprocess the data by dropping irrelevant columns,
        handling missing values, and encoding categorical columns.
        """
        logger.info("Preprocessing the data")
        try:
            # Drop irrelevant columns
            data = data.drop(columns=['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId'])

            # Convert categorical columns to one-hot encoding
            categorical_columns = ['CurrencyCode', 'CountryCode', 'ProductId', 'ChannelId']
            data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

            # Handle 'TransactionStartTime' by extracting useful datetime features
            if 'TransactionStartTime' in data.columns:
                logger.info("Extracting datetime features from 'TransactionStartTime'")
                data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
                data['TransactionHour'] = data['TransactionStartTime'].dt.hour
                data['TransactionDay'] = data['TransactionStartTime'].dt.day
                data['TransactionMonth'] = data['TransactionStartTime'].dt.month
                data['TransactionWeekday'] = data['TransactionStartTime'].dt.weekday
                data = data.drop(columns=['TransactionStartTime'])

            logger.info("Data preprocessing completed")
            return data
        except Exception as e:
            logger.error(f"Error occurred while preprocessing the data: {e}")
            return None

    def split_features_target(self, data, target_column='Label'):
        """
        Split the data into features and target.
        
        Parameters:
        - data: DataFrame containing features and target
        - target_column: The name of the target column
        
        Returns:
        - X: DataFrame containing features
        - y: Series containing target
        """
        X = data.drop(columns=[target_column])
        y = data[target_column]
        return X, y

    def split_the_data(self, data):
        """
        Split the data into training and testing sets.
        
        Parameters:
        - data: DataFrame containing features and target
        
        Returns:
        - X_train: Training data features
        - X_test: Testing data features
        - y_train: Training data target
        - y_test: Testing data target
        """
        logger.info("Splitting the data")
        try:
            X = data.drop(columns=['Label'])  # Features
            y = data['Label']  # Target variable
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error occurred while splitting the data: {e}")
            return None, None, None, None

    def tain_the_models(X_train,y_train,X_test):
        """
        Train Logistic Regression and Random Forest models.
        
        Returns:
        - logistic_model: Trained Logistic Regression model
        - random_forest_model: Trained Random Forest model
        """
        logger.info("Training the models")
        try:
            # Scale the training data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
           
            logger.info("Initializing the models")
            # Initialize the models
            logistic_model = LogisticRegression(max_iter=1000, random_state=42)
            random_forest_model = RandomForestClassifier(random_state=42)
            
            logger.info("Training the models with our data")
            # Train the models 
            logistic_model.fit(X_train_scaled, y_train)
            random_forest_model.fit(X_train, y_train)
            
            return logistic_model, random_forest_model
        except Exception as e:
            logger.error(f"Error occurred while training the models: {e}")
            return None, None

    def evaluate_models(self, model, X_test, y_test):
        """
        Evaluate the given model.
        
        Parameters:
        - model: Trained model to evaluate
        - X_test: Test data features
        - y_test: Test data target
        
        Returns:
        - y_pred: Predictions made by the model
        """
        logger.info("Evaluating the model")
        try:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")

            return y_pred
        except Exception as e:
            logger.error(f"Error occurred while evaluating the model: {e}")
            return None
