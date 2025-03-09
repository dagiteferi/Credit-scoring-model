import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

# Import the functions from your credit scoring pipeline script
# Assuming the script is saved as 'credit_scoring_pipeline.py'
from credit_scoring_pipeline import (
    preprocess_data,
    split_the_data,
    define_models,
    define_hyperparameter_grids,
    perform_grid_search,
    evaluate_best_models,
    save_best_models
)

# Suppress logging output during tests
logging.getLogger().setLevel(logging.CRITICAL)

class TestCreditScoringPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Create a sample dataset to use across all tests."""
        cls.data = pd.DataFrame({
            'TransactionId': [1, 2, 3, 4],
            'BatchId': [101, 102, 103, 104],
            'AccountId': [1001, 1002, 1003, 1004],
            'SubscriptionId': [2001, 2002, 2003, 2004],
            'CustomerId': [3001, 3002, 3003, 3004],
            'CurrencyCode': ['USD', 'EUR', 'USD', 'EUR'],
            'CountryCode': ['US', 'FR', 'US', 'FR'],
            'ProductId': [1, 2, 1, 2],
            'ChannelId': [1, 2, 1, 2],
            'TransactionStartTime': ['2023-01-01 12:00:00', '2023-01-02 13:00:00', 
                                    '2023-01-03 14:00:00', '2023-01-04 15:00:00'],
            'Amount': [100.0, 200.0, np.nan, 400.0],
            'Label': [0, 1, 0, 1]
        })
        cls.data['TransactionStartTime'] = pd.to_datetime(cls.data['TransactionStartTime'])

    def test_preprocess_data(self):
        """Test the preprocess_data function."""
        processed_data = preprocess_data(self.data.copy())
        self.assertIsNotNone(processed_data, "Preprocessing returned None")
        self.assertFalse(processed_data.isnull().any().any(), "Processed data contains missing values")
        self.assertNotIn('TransactionId', processed_data.columns, "TransactionId was not dropped")
        self.assertIn('TransactionHour', processed_data.columns, "TransactionHour feature not created")
        self.assertIn('CurrencyCode_EUR', processed_data.columns, "Categorical encoding failed for CurrencyCode")

    def test_split_the_data(self):
        """Test the split_the_data function."""
        processed_data = preprocess_data(self.data.copy())
        X_train, X_test, y_train, y_test = split_the_data(processed_data, target_column='Label', test_size=0.2)
        self.assertIsNotNone(X_train, "X_train is None")
        self.assertIsNotNone(X_test, "X_test is None")
        self.assertIsNotNone(y_train, "y_train is None")
        self.assertIsNotNone(y_test, "y_test is None")
        self.assertEqual(len(X_train), int(0.8 * len(processed_data)), "Training set size incorrect")
        self.assertEqual(len(X_test), int(0.2 * len(processed_data)), "Test set size incorrect")

    def test_define_models(self):
        """Test the define_models function."""
        models = define_models()
        self.assertIn('LogisticRegression', models, "LogisticRegression not in models")
        self.assertIn('RandomForest', models, "RandomForest not in models")
        self.assertIsInstance(models['LogisticRegression'], Pipeline, "LogisticRegression is not a Pipeline")
        self.assertIsInstance(models['RandomForest'], RandomForestClassifier, "RandomForest is not a RandomForestClassifier")

    def test_define_hyperparameter_grids(self):
        """Test the define_hyperparameter_grids function."""
        param_grids = define_hyperparameter_grids()
        self.assertIn('LogisticRegression', param_grids, "LogisticRegression not in param_grids")
        self.assertIn('RandomForest', param_grids, "RandomForest not in param_grids")
        self.assertIn('logistic__C', param_grids['LogisticRegression'], "C parameter missing for LogisticRegression")
        self.assertIn('n_estimators', param_grids['RandomForest'], "n_estimators missing for RandomForest")

    @patch('sklearn.model_selection.GridSearchCV')
    def test_perform_grid_search(self, mock_grid_search):
        """Test the perform_grid_search function with mocking."""
        # Mock GridSearchCV to avoid actual fitting
        mock_instance = MagicMock()
        mock_instance.best_estimator_ = LogisticRegression()
        mock_grid_search.return_value = mock_instance
        models = define_models()
        param_grids = define_hyperparameter_grids()
        processed_data = preprocess_data(self.data.copy())
        X_train, _, y_train, _ = split_the_data(processed_data)
        best_models = perform_grid_search(models, param_grids, X_train, y_train)
        self.assertIsNotNone(best_models, "perform_grid_search returned None")
        self.assertIn('LogisticRegression', best_models, "LogisticRegression missing from best_models")
        self.assertIn('RandomForest', best_models, "RandomForest missing from best_models")

    def test_evaluate_best_models(self):
        """Test the evaluate_best_models function."""
        # Prepare dummy data and models
        processed_data = preprocess_data(self.data.copy())
        X_train, X_test, y_train, y_test = split_the_data(processed_data)
        best_models = {
            'LogisticRegression': Pipeline([
                ('scaler', StandardScaler()),
                ('logistic', LogisticRegression(solver='liblinear', random_state=42))
            ]).fit(X_train, y_train),
            'RandomForest': RandomForestClassifier(random_state=42).fit(X_train, y_train)
        }
        results = evaluate_best_models(best_models, X_test, y_test)
        self.assertIsNotNone(results, "evaluate_best_models returned None")
        self.assertIn('LogisticRegression', results, "LogisticRegression missing from results")
        self.assertIn('RandomForest', results, "RandomForest missing from results")
        self.assertIn('Accuracy', results['LogisticRegression'], "Accuracy missing for LogisticRegression")

    @patch('joblib.dump')
    @patch('os.makedirs')
    def test_save_best_models(self, mock_makedirs, mock_joblib_dump):
        """Test the save_best_models function with mocking."""
        best_models = {
            'LogisticRegression': LogisticRegression(),
            'RandomForest': RandomForestClassifier()
        }
        save_best_models(best_models, save_dir='test_models')
        mock_makedirs.assert_called_once_with('test_models')
        self.assertEqual(mock_joblib_dump.call_count, 2, "joblib.dump not called twice")

if __name__ == '__main__':
    unittest.main()