# Understanding Credit Scores and Microloans

Credit scores are numerical representations that gauge how likely an individual is to repay a loan and make timely payments. Essentially, they predict one's credit behavior—such as the likelihood of paying back loans on time—using information from credit reports.loans typically given to borrowers who lack collateral. These financial tools provide a crucial lifeline for those with limited access to traditional lending resources.
# Role of Data in Credit Scoring

Data play an important role in the development, monitoring, and maintenance of credit scoring models. By utilizing data, these models gain higher predictive power and deeper insights, which allow financial institutions to offer new products to previously excluded sectors of society, granting them access to affordable credit.
### Types of Data Used for Credit Scoring

Traditionally, credit data are utilized for credit scoring, which include the following:

- **Amount of Loan**: The total sum borrowed.
- **Type of Loan**: Categories such as personal, mortgage, auto, etc.
- **Maturity of Loan**: The duration until the loan is fully repaid.
- **Guarantees and Collateral Value**: Assets pledged to secure the loan.
- **Historical Payment Performance**: Records of default information and payments in arrears.
- **Amounts Owed**: The current outstanding debt.
- **Length of Credit History**: The duration of a borrower's credit activity.
- **New Credit**: Recent credit inquiries and accounts.
- **Types of Credit**: The variety of credit accounts held, such as credit cards, retail accounts, installment loans, etc.

These data points are factored into a credit score as indicators of the borrower’s willingness and ability to repay their debts.
### Table 1: Types of Data Used for Credit Scoring

| Data Category | Data Type                   | Credit Scoring Application                                                                                               |
|---------------|-----------------------------|-------------------------------------------------------------------------------------------------------------------------|
| Traditional   | Bank transactional data     | Records of late payments on current and past credit, current loan amounts and loan purpose, credit history               |
| Traditional   | Credit bureau checks        | Number of credit inquiries                                                                                              |
| Traditional   | Commercial data             | Financial statements, number of working capital loans, and others                                                       |
| Alternative   | Utilities data              | Steady records of on-time payments as a possible indicator of creditworthiness                                          |
| Alternative   | Social media                | Social media data providing insights on consumer’s lifestyle                                                            |
| Alternative   | Credit scoring application  | Mobile applications                                                                                                     |
| Alternative   | Online transactions         | Data from online transactions                                                                                           |
| Alternative   | Behavioral data             | Mobile payment systems providing insights into the consumer’s behavior                                                  |
| Alternative   | Granular transactional data | Detailed insights into spending patterns                                                                                |
| Alternative   | Psychometrics               | Form filling data                                                                                                       |

### Modern Credit Scoring Systems

Modern credit scoring systems can collate data from a wide variety of sources in structured, unstructured, and semi-structured forms. New data sources include granular spending behavior, mobile data, geolocation data, and payment data from utilities and, in some cases, social media.

#### Mobile Data
Mobile applications may collect data such as transport movements, geolocation, and transactional data. This data allows mobile phone applications to perform credit checks that traditional credit scoring providers (CSPs) may find challenging. However, data subjects may be unaware that their personal data is used for credit scoring.

#### Social Media Data
Research suggests that the number and frequency of social media posts can provide insights into consumers' lifestyles, expenditures, and willingness to repay debt.


# Credit Scoring Methodologies

### Traditional Credit Scoring Methods
The most prominent techniques used to develop credit scorecards include statistical discrimination and classification methods such as linear regression models, discriminant analysis, logit and probit models, and expert judgment-based models.

#### Linear Regression
Linear regression is useful in credit scoring due to its ease in explaining and predicting risk parameters, like the probability of default.

#### Discriminant Analysis
Discriminant analysis, a variation of regression analysis, is used for classification based on categorical data, such as "default" versus "nondefault".

#### Probit Analysis and Logistic Regression
The logit model is popular for estimating the probability of default. It is easy to develop, validate, calibrate, and interpret. Estimation in logistic regression maximizes the likelihood of observing the sample values.

#### Judgment-Based Models
Judgment-based models, like the Analytic Hierarchy Process (AHP), rely on human judgments to perform evaluations. These models are crucial for evaluating exceptions and underrepresented instances in the data.
# Machine Learning in Credit Scoring

## Techniques Related to Credit Scoring Methods

### Feature Engineering
Feature engineering involves transforming the input data set into meaningful features that help to reveal the underlying patterns in the data. This process enhances the accuracy of machine learning models by providing them with more relevant and informative data.

### Reinforcement Learning
Reinforcement learning enables machines to learn behaviors based on feedback from the environment. This type of learning allows the model to adapt its actions to achieve optimal outcomes, either through a one-time learning process or continuous adaptation over time.
### Development of Machine Learning Models for Default Prediction

Developing a machine learning model for default prediction involves three phases:

1. **Preparation Phase**: Selects suitable machine learning algorithms and explores alternative data for credit scoring.
2. **Training Phase**: Trains the model using the selected algorithms and pre-processed data.
3. **Evaluation Phase**: Assesses the model's results, considers the outputs, and ensures model interpretability.

### Model Selection
Model selection is an exploratory process involving the continuous evaluation of multiple machine learning models. Some commonly used algorithms include Logistic Regression, Random Forest, Extra-Trees, CatBoost, LightGBM, XGBoost, K-Nearest Neighbours, Convolutional Neural Networks, and Stacking. (Refer to Appendix B for detailed descriptions of these algorithms.)

### Data Exploration
Credit scoring modelling requires rigorous exploratory data analysis (EDA) to distinguish variables and optimize the model.

### Model Training
Machine learning algorithms establish statistical/mathematical models that can make inferences. The training data (predictors/independent variables) and the outputs (responses/dependent variables) can be quantitative (numerical) or qualitative (categorical). Numerical outputs correspond to regression problems, while categorical outputs correspond to classification problems.

### Model Assessment
In the banking industry, model quality resides in the output and its interpretation. The model must be accurate to avoid losses and align with the bank's working capacity and liquidity. The Area Under the Receiver Operating Characteristic (ROC) Curve is a commonly used metric to assess model quality.

### Feature Importance
Investigating feature importance helps to further improve the model. Once the data has been explored and the model trained and tested, analyzing feature importance involves inspecting variables and assessing whether changes in a variable would affect the model output.

### Model Interpretability
Interpretability is the degree to which a human can understand the cause of a decision. Greater interpretability makes it easier for humans to understand why a decision or prediction was made. Model interpretation technologies include SHAP, ELI, LIME, Microsoft InterpretML, XAI, Alibi, TreeInterpreter, Skater, FairML, and fairness.
