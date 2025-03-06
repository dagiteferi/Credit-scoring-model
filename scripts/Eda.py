import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logger for the module
logger = logging.getLogger(__name__)



def data_overview(data):
    """
    Print the structure of the dataset.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to analyze.
    """
    print("Number of rows and columns:", data.shape)
    print("Column names and data types:\n", data.dtypes)

def summary_statistics(data):
    """
    Display summary statistics of the dataset.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to analyze.
    """
    print("Summary statistics:\n", data.describe())

def visualize_distribution(data, numeric_cols):
    """
    Visualize the distribution of numerical features with histograms.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to analyze.
    numeric_cols : list
        List of numerical column names to visualize.
    """
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(3, 3, i)  # Adjust layout based on number of columns
        plt.hist(data[col], bins=30, edgecolor='black')
        plt.title(col)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def box_plot_for_outliers(data, numeric_cols):
    """
    Create box plots for outlier detection in numerical features.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to analyze.
    numeric_cols : list
        List of numerical column names to visualize.
    """
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(3, 3, i)  # Adjust layout based on number of columns
        sns.boxplot(data[col])
        plt.title(col)
        plt.ylabel('Value')
    plt.tight_layout()
    plt.show()

def pair_plots(data, numeric_cols):
    """
    Create pair plots for multivariate analysis of numerical features.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to analyze.
    numeric_cols : list
        List of numerical column names to include in pair plots.
    """
    sns.pairplot(data[numeric_cols])
    plt.show()

def check_for_skewness(data, numeric_cols):
    """
    Check and visualize skewness in numerical features.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to analyze.
    numeric_cols : list
        List of numerical column names to check.
    """
    skewness = data[numeric_cols].skew()
    print("Skewness for Numerical Features:\n", skewness)
    plt.figure(figsize=(10, 5))
    skewness.plot(kind='bar')
    plt.title('Skewness of Numerical Features')
    plt.xlabel('Features')
    plt.ylabel('Skewness')
    plt.axhline(0, color='red', linestyle='--')
    plt.show()

def distribution_of_numerical_features(data):
    """
    Perform and visualize the distribution analysis of numerical features.

    This includes histograms, box plots for outliers, pair plots, and skewness checks.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to analyze.

    Raises
    ------
    Exception
        If an error occurs during the distribution analysis.
    """
    logger.info("Starting distribution analysis of numerical features")
    try:
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        logger.info(f"Identified {len(numeric_cols)} numerical columns: {numeric_cols}")
        
        logger.info("Generating histograms")
        visualize_distribution(data, numeric_cols)
        
        logger.info("Generating box plots for outlier detection")
        box_plot_for_outliers(data, numeric_cols)
        
        logger.info("Generating pair plots for multivariate analysis")
        pair_plots(data, numeric_cols)
        
        logger.info("Checking for skewness")
        check_for_skewness(data, numeric_cols)
        
        logger.info("Distribution analysis completed")
    except Exception as e:
        logger.error(f"An error occurred during distribution analysis: {e}")
        raise e

def correlation_analysis(data):
    """
    Plot a correlation matrix heatmap for numerical features.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to analyze.
    """
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = data[numeric_cols].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()

def identify_missing_values(data):
    """
    Identify and print missing values in the dataset.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to analyze.
    """
    missing_values = data.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        print("Missing values in each column:\n", missing_values)
    else:
        print("No missing values found.")