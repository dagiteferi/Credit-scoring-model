import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

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
       
    def summary_statistics(self):
        """Display summary statistics of the dataset."""
        print("Summary statistics:\n", self.data.describe())

    def visualize_distribution(self, numeric_cols):
        """Visualize the distribution of numerical features."""
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(3, 3, i)  # Adjust the layout based on the number of numerical columns
            plt.hist(self.data[col], bins=30, edgecolor='black')
            plt.title(col)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    def box_plot_for_outliers_detections(self, numeric_cols):
        """Create box plots for outliers detection."""
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(3, 3, i)  # Adjust the layout based on the number of numerical columns
            sns.boxplot(self.data[col])
            plt.title(col)
            plt.ylabel('Value')
        plt.tight_layout()
        plt.show()

    def pair_plots_for_multivariate_analysis(self, numeric_cols):
        """Create pair plots for multivariate analysis."""
        sns.pairplot(self.data[numeric_cols])
        plt.show()

    def check_for_skewness(self, numeric_cols):
        """Check for skewness in numerical features."""
        skewness = self.data[numeric_cols].skew()
        print("Skewness for Numerical Features:\n", skewness)
        # Visualize skewness with a bar plot
        plt.figure(figsize=(10, 5))
        skewness.plot(kind='bar')
        plt.title('Skewness of Numerical Features')
        plt.xlabel('Features')
        plt.ylabel('Skewness')
        plt.axhline(0, color='red', linestyle='--')
        plt.show()

    def distribution_of_numerical_features(self):
        """Perform the process to show the distribution of numerical features."""
        logger.info("Performing the process to show the distribution of numerical features")
        try:
            logger.info("Selecting columns of the numeric columns only")
            numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
            logger.info("Visualizing the numeric data")
            self.visualize_distribution(numeric_cols)
            logger.info("Box plot for outliers for numeric columns")
            self.box_plot_for_outliers_detections(numeric_cols)
            logger.info("Pair plots for multivariate analysis")
            self.pair_plots_for_multivariate_analysis(numeric_cols)
            logger.info("Checking for skewness in numeric columns")
            self.check_for_skewness(numeric_cols)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
    
    def correlation_analysis(self):
        """Understand the relationship between numerical features."""
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        correlation_matrix = self.data[numeric_cols].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.show()
    
    def identify_missing_values(self):
        """Identify missing values in the dataset."""
        missing_values = self.data.isnull().sum()
        missing_values = missing_values[missing_values > 0]  # Filter only columns with missing values
        print("Missing values in each column:\n", missing_values)
    plt.show()

    def plot_outliers(self):
        """Use box plots to identify outliers."""
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(3, 3, i)  # Adjust the layout based on the number of numerical columns
            sns.boxplot(self.data[col])
            plt.title(col)
            plt.ylabel('Value')
        plt.tight_layout()
        plt.show()

# Initialize logger
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# Example usage
path = '../data/data.csv'
eda = CreditScoringEDA(path)

# Call the data_overview method to get an overview
eda.data_overview()

# Call the identify_missing_values method to detect missing values
eda.identify_missing_values()
