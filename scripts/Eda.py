import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class CreditScoringEDA:
    
    def __init__(self, file_path):
        self.data = self.load_data(file_path)
    
    def load_data(self, file_path):
        """Load the dataset."""
        data = pd.read_csv(file_path)
        return data

    def data_overview(self):
        """Print the structure of the dataset."""
        print("Number of rows and columns:", self.data.shape)
        print("Column names and data types:\n", self.data.dtypes)
        print("First few rows of the dataset:\n", self.data.head())

    def summary_statistics(self):
        """Display summary statistics of the dataset."""
        print("Summary statistics:\n", self.data.describe())

    def plot_numerical_distribution(self, numerical_columns):
        """Visualize the distribution of numerical features."""
        for column in numerical_columns:
            plt.figure(figsize=(10, 5))
            sns.histplot(self.data[column], kde=True)
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()

    def plot_categorical_distribution(self, categorical_columns):
        """Analyze the distribution of categorical features."""
        for column in categorical_columns:
            plt.figure(figsize=(10, 5))
            sns.countplot(self.data[column])
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.xticks(rotation=90)
            plt.show()

    def correlation_analysis(self):
        """Understand the relationship between numerical features."""
        correlation_matrix = self.data.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.show()

    def identify_missing_values(self):
        """Identify missing values in the dataset."""
        missing_values = self.data.isnull().sum()
        print("Missing values in each column:\n", missing_values)

    def plot_outliers(self, numerical_columns):
        """Use box plots to identify outliers."""
        for column in numerical_columns:
            plt.figure(figsize=(10, 5))
            sns.boxplot(x=self.data[column])
            plt.title(f'Box plot of {column}')
            plt.xlabel(column)
            plt.show()

