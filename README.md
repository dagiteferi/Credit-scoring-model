# Credit Scoring Model for Bati Bank

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [API Development](#api-development)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction
In a world where financial inclusion is critical, providing underserved communities with access to credit becomes paramount. This project aims to develop a comprehensive and effective credit scoring system for Bati Bank, leveraging data from an eCommerce platform to enable a buy-now-pay-later service for customers. The journey encompasses understanding credit scoring methodologies, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

## Project Structure
```plaintext
dagiteferi-credit-scoring-model/
├── README.md
├── requirements.txt
├── credit_scoring/
│   ├── db.sqlite3
│   ├── manage.py
│   ├── urls.py
│   ├── credit_scoring/
│   │   ├── __init__.py
│   │   ├── asgi.py
│   │   ├── settings.py
│   │   ├── urls.py
│   │   ├── views.py
│   │   ├── wsgi.py
│   │   └── templates/
│   │       └── predict.html
│   ├── scoring_api/
│   │   ├── __init__.py
│   │   ├── admin.py
│   │   ├── apps.py
│   │   ├── models.py
│   │   ├── serializers.py
│   │   ├── tests.py
│   │   ├── urls.py
│   │   ├── views.py
│   │   └── migrations/
│   │       └── __init__.py
│   └── webapp/
│       ├── __init__.py
│       ├── admin.py
│       ├── apps.py
│       ├── models.py
│       ├── tests.py
│       ├── urls.py
│       ├── views.py
│       ├── migrations/
│       │   └── __init__.py
│       └── templates/
│           └── home.html
├── models/
│   ├── Logistic Regression_best_model.pkl
│   └── Random Forest_best_model.pkl
├── notebooks/
│   ├── README.md
│   ├── Exploratory_Data_Analysis.ipynb
│   ├── Feature_Engineering.ipynb
│   ├── Modelling.ipynb
│   └── __init__.py
├── scripts/
│   ├── README.md
│   ├── Eda.py
│   ├── Feature_Engineering.py
│   ├── Modelling.py
│   ├── __init__.py
│   └── file_structure.py
├── src/
│   └── __init__.py
├── tests/
│   ├── __init__.py
│   └── test_api.py
└── .github/
    └── workflows/
        └── unittests.yml
```
## Installation
Clone the repository
```bash
git clone https://github.com/your-repo/dagiteferi-credit-scoring-model.git
```
Navigate to the project directory
```bash
cd dagiteferi-credit-scoring-model
```
Create a virtual environment
```bash
python3 -m venv venv
```
Activate the virtual environment
```bash
# On Windows
venv\Scripts\activate
```
```bash
# On macOS/Linux
source venv/bin/activate
```

Install the required dependencies
```bash
pip install -r requirements.txt
```
## Usage
Running the Django Application
Navigate to the credit_scoring directory
```bash
cd credit_scoring
```
Apply the database migrations
```bash
python manage.py migrate
```
Run the Django development server
```bash
python manage.py runserver
```

Making Predictions
The API endpoint /api/predict/ can be used to make predictions. Use the following sample payload:
```json
{
    "ProviderId": 4,
    "ProductCategory": 2,
    "Amount": 0.050426,
    "Value": 0.076352,
    "PricingStrategy": 2,
    "FraudResult": 0,
    "Total_Transaction_Amount": 0.165893,
    "Average_Transaction_Amount": -0.074327,
    "Transaction_Count": -1,
    "Std_Transaction_Amount": 0.145069,
    "Transaction_Hour": 19,
    "Transaction_Day": 15,
    "Transaction_Month": 12,
    "Transaction_Year": 2025,
    "CurrencyCode_WOE": 0.0,
    "ProviderId_WOE": 3.137005,
    "ProductId_WOE": 1.645067,
    "ProductCategory_WOE": 1.620379,
    "Recency": 2265,
    "RFMS_score": -0.042337,
    "ProductId_1": false,
    "ProductId_2": false,
    "ProductId_3": false,
    "ProductId_4": false,
    "ProductId_5": false,
    "ProductId_6": false,
    "ProductId_7": false,
    "ProductId_8": false,
    "ProductId_9": false,
    "ProductId_10": false,
    "ProductId_11": false,
    "ProductId_12": false,
    "ProductId_13": false,
    "ProductId_14": false,
    "ProductId_15": false,
    "ProductId_16": false,
    "ProductId_17": false,
    "ProductId_18": false,
    "ProductId_19": false,
    "ProductId_20": false,
    "ProductId_21": false,
    "ProductId_22": false,
    "ChannelId_ChannelId_2": true,
    "ChannelId_ChannelId_3": false,
    "ChannelId_ChannelId_5": true,
    "TransactionHour": 15,
    "TransactionDay": 11,
    "TransactionMonth": 5,
    "TransactionWeekday": 1,
    "ProductId": 1
}

```
## Features
Exploratory Data Analysis (EDA): Understand data distributions, identify outliers, and visualize correlations.

Feature Engineering: Create aggregate features, extract new features, encode categorical variables, handle missing values, normalize numerical features, and calculate Weight of Evidence (WoE).

Model Training and Evaluation: Train multiple machine learning models, perform hyperparameter tuning, evaluate models, and save the best models.

API Development: Develop a REST API for real-time credit scoring predictions.


## Exploratory Data Analysis (EDA)
The EDA process involves the following steps:

Data Loading and Overview: Load and inspect the dataset.

Missing Values Identification: Check for missing values.

Numerical Features Distribution: Analyze the distribution of numerical features.

Skewness Analysis: Check for skewness in numerical features.

Correlation Analysis: Understand the relationships between features.

Outlier Detection: Identify and handle outliers.


## Feature Engineering
Feature engineering involves several tasks:

Create Aggregate Features: Summarize transaction data for each customer.

Extract Features: Derive new features from existing data.

Encode Categorical Variables: Convert categorical variables into numerical format.

Handle Missing Values: Ensure the dataset is clean.

Normalize/Standardize Numerical Features: Enhance model performance.

Construct RFMS Scores and Labels: Calculate RFMS scores and assign labels.

Calculating Weight of Evidence (WoE): Transform categorical variables

