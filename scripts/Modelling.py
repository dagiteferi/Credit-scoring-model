from sklearn.model_selection import train_test_split

class Modeling:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()

    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.
        
        Parameters:
        - test_size: float, optional (default=0.2)
            The proportion of the dataset to include in the test split.
        - random_state: int, optional (default=42)
            Controls the shuffling applied to the data before the split.
        
        Returns:
        - X_train: Training data features
        - X_test: Testing data features
        - y_train: Training data target
        - y_test: Testing data target
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
