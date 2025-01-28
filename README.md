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




