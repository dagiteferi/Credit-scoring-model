# Credit Scoring Model for Bati Bank

![Credit Scoring Banner](https://via.placeholder.com/1200x300.png?text=Credit+Scoring+Model+for+Bati+Bank)  
*Empowering Financial Inclusion with AI-Driven Credit Risk Assessment*

---

## 📖 Table of Contents

- [Introduction](#introduction)
- [✨ Features](#-features)
- [📂 Project Structure](#-project-structure)
- [⚙️ Installation](#️-installation)
- [🚀 Usage](#-usage)
  - [Running the Backend](#running-the-backend)
  - [Using the Frontend](#using-the-frontend)
  - [Making API Predictions](#making-api-predictions)
- [🔍 Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [🛠️ Feature Engineering](#️-feature-engineering)
- [🤖 Model Training and Evaluation](#-model-training-and-evaluation)
- [🔮 Model Explainability](#-model-explainability)
- [🌐 API Development](#-api-development)
- [💻 Frontend Interface](#-frontend-interface)
- [🚀 Deployment](#-deployment)
- [🎯 Challenges and Solutions](#-challenges-and-solutions)
- [🔮 Future Work](#-future-work)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## Introduction

The **Credit Scoring Model for Bati Bank** is an AI-powered platform designed to assess credit risk using eCommerce transaction data. This solution enables financial inclusion through:

- **📈 Accurate Predictions**: Random Forest model achieves **ROC-AUC: 0.9998**
- **🔍 Transparent Decisions**: SHAP explanations and feature importance visualizations
- **⚡ Real-Time Processing**: FastAPI backend with <100ms response times
- **📱 Mobile-First Interface**: Responsive design accessible on all devices

---

## ✨ Features

- **Automated Data Pipelines**
  - RFMS scoring (Recency, Frequency, Monetary, Score)
  - WoE encoding for categorical features
- **Advanced Modeling**
  - Hyperparameter-tuned Random Forest & Logistic Regression
  - Cross-validation with stratified sampling
- **Production-Ready Deployment**
  - Dockerized environment
  - CI/CD pipeline with GitHub Actions
- **User-Centric Interface**
  - Dual form system (Quick/Detailed assessment)
  - Interactive risk visualization dashboard

---

## 📂 Project Structure

~~~bash
dagiteferi-credit-scoring-model/
├── 📁 credit_scoring_app/       # FastAPI backend
├── 📁 models/                   # Serialized ML models
├── 📁 notebooks/                # Jupyter analysis notebooks
├── 📁 scripts/                  # Data processing scripts
├── 📁 static/                   # CSS/JS assets
└── 📁 tests/                    # Unit/integration tests
~~~

---

## ⚙️ Installation

```bash
git clone https://github.com/your-repo/dagiteferi-credit-scoring-model.git
cd dagiteferi-credit-scoring-model
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt

## 🚀 Usage

### Running the Backend
```bash
cd credit_scoring_app
uvicorn main:app --host 0.0.0.0 --port 8000
```
### Using the Frontend
Access at http://localhost:8000/static/index.html

![image](https://github.com/user-attachments/assets/6cfda001-0733-40d2-80a7-6a0b1f563268)
#### Making API Predictions
```bash
curl -X POST "http://localhost:8000/predict/good" \
-H "Content-Type: application/json" \
-d '{
  "TransactionId": 1,
  "Amount": 0.05,
  "FraudResult": 0
}'
```
## 🔍 Exploratory Data Analysis (EDA)

**Key Insights:**  
- 🎯 **Class Imbalance**: Only 0.2% fraud cases  
- 📉 **Skewed Distributions**: Transaction amounts follow power law  
- 🔗 **Strong Correlations**:  
  - `RFMS_score` ↔ `Total_Transaction_Amount` (ρ=0.89)  
  - `Transaction_Count` ↔ `Product_Variety` (ρ=0.76)  

---

## 🛠️ Feature Engineering

**Transformations Applied:**  
1. **Temporal Features**  
   - Transaction hour/day/month  
   - Time since last transaction  
2. **Aggregate Features**  
   - 30-day rolling transaction count  
   - Customer lifetime value  

---

## 🤖 Model Training and Evaluation

| Model               | ROC-AUC | Precision | Recall | F1-Score |
|---------------------|---------|-----------|--------|----------|
| Random Forest       | 0.9998  | 0.997     | 0.998  | 0.997    |
| Logistic Regression | 0.9962  | 0.982     | 0.961  | 0.971    |

---

## 🔮 Model Explainability

**SHAP Analysis:**  
- **Top Predictive Features**:  
  1. `Total_Transaction_Amount` (SHAP value: 1.42)  
  2. `RFMS_score` (SHAP value: 1.18)  
  3. `Transaction_Recency` (SHAP value: 0.76)  

---

## 🌐 API Development

**Endpoints:**  
```python
@app.post("/predict/good")
async def predict_good_risk(data: CustomerData):
    return predict(data, model_path="models/RandomForest_best_model.pkl")
