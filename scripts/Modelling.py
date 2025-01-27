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
            data = pd.read_csv(path)
            return data
        except Exception as e:
            logger.error(f"Error occurred while loading the data: {e}")
            return None

    def preprocess_data(self, data):
        logger.info("Preprocessing the data")
        try:
            data = data.drop(columns=['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId'])
            categorical_columns = ['CurrencyCode', 'CountryCode', 'ProductId', 'ChannelId']
            data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
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
        X = data.drop(columns=[target_column])
        y = data[target_column]
        return X, y

    def split_the_data(self, data):
        logger.info("Splitting the data")
        try:
            X = data.drop(columns=['Label'])
            y = data['Label']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error occurred while splitting the data: {e}")
            return None, None, None, None

    def train_the_models(self, X_train, y_train):
        logger.info("Training the models")
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            logistic_model = LogisticRegression(max_iter=1000, random_state=42)
            random_forest_model = RandomForestClassifier(random_state=42)
            logistic_model.fit(X_train_scaled, y_train)
            random_forest_model.fit(X_train, y_train)
            return logistic_model, random_forest_model
        except Exception as e:
            logger.error(f"Error occurred while training the models: {e}")
            return None, None

    def evaluate_models(self, model, X_test, y_test):
        logger.info("Evaluating the model")
        try:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=1)
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


