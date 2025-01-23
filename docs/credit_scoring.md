# Understanding Credit Scores and Microloans
Credit scores are numerical representations that gauge how likely an individual is to repay a loan and make timely payments. Essentially, they predict one's credit behavior—such as the likelihood of paying back loans on time—using information from credit reports. Microloans are typically given to borrowers who lack collateral, providing a crucial lifeline for those with limited access to traditional lending resources.

### Applications of Credit Scoring

1. **Personal Loans**
   - Credit scoring helps lenders assess the creditworthiness of individuals applying for personal loans. A higher credit score can lead to better loan terms, such as lower interest rates.

2. **Mortgages**
   - Credit scores are crucial in the mortgage approval process. Lenders use credit scores to determine the risk of lending to a borrower and to set the interest rate for the mortgage.

3. **Credit Cards**
   - Issuers of credit cards use credit scores to decide whether to approve an application and to set the credit limit and interest rates for the cardholder.

4. **Auto Loans**
   - Credit scores are used by auto lenders to evaluate the risk of lending money for vehicle purchases. Higher credit scores can result in better financing options.

5. **Business Loans**
   - Credit scores are important for business loan applications, helping lenders assess the creditworthiness of small and medium-sized enterprises (SMEs).

6. **Insurance Premiums**
   - Some insurance companies use credit scores to determine the premiums for auto, home, and other types of insurance policies. 

7. **Rental Applications**
   - Landlords and property management companies may use credit scores to evaluate the reliability of potential tenants.

8. **Utility Services**
   - Utility companies may use credit scores to decide whether to require a deposit from new customers or to set terms for utility services.

9. **Employment Screening**
   - Some employers use credit scores as part of their hiring process, especially for positions that involve financial responsibilities.

10. **Telecommunications Services**
    - Telecom companies may use credit scores to determine eligibility for postpaid mobile phone plans and to set terms for service contracts.
### Benefits of Credit Scoring Models

1. **Improved Risk Assessment**
   - Credit scoring models provide lenders with a more accurate assessment of a borrower's creditworthiness, reducing the risk of default and financial loss.

2. **Faster Decision-Making**
   - Automated credit scoring models enable quicker loan approval processes, benefiting both lenders and borrowers by reducing the time required for credit evaluation.

3. **Objective Evaluation**
   - Credit scoring models use quantitative data to make decisions, minimizing human biases and ensuring fair and consistent evaluations.

4. **Cost Efficiency**
   - Automated credit scoring reduces the need for manual review, lowering operational costs for lenders.

5. **Increased Access to Credit**
   - By utilizing alternative data sources, credit scoring models can extend credit to individuals with limited or no credit history, promoting financial inclusion.

6. **Customization of Loan Products**
   - Credit scoring models allow lenders to tailor loan products and interest rates to individual borrowers based on their credit risk, offering more personalized financial solutions.

7. **Enhanced Monitoring and Management**
   - Ongoing monitoring of credit scores enables lenders to identify potential risks early and manage their loan portfolios more effectively.

8. **Regulatory Compliance**
   - Credit scoring models help financial institutions comply with regulatory requirements by providing documented and transparent credit evaluation processes.

## Role of Data in Credit Scoring
Data play an important role in the development, monitoring, and maintenance of credit scoring models. By utilizing data, these models gain higher predictive power and deeper insights, allowing financial institutions to offer new products to previously excluded sectors of society, granting them access to affordable credit.

## Types of Data Used for Credit Scoring
Traditionally, credit data are utilized for credit scoring, including:

- **Amount of Loan:** The total sum borrowed.
- **Type of Loan:** Categories such as personal, mortgage, auto, etc.
- **Maturity of Loan:** The duration until the loan is fully repaid.
- **Guarantees and Collateral Value:** Assets pledged to secure the loan.
- **Historical Payment Performance:** Records of default information and payments in arrears.
- **Amounts Owed:** The current outstanding debt.
- **Length of Credit History:** The duration of a borrower's credit activity.
- **New Credit:** Recent credit inquiries and accounts.
- **Types of Credit:** The variety of credit accounts held, such as credit cards, retail accounts, installment loans, etc.

These data points are factored into a credit score as indicators of the borrower’s willingness and ability to repay their debts.

## Table 1: Types of Data Used for Credit Scoring

| Data Category  | Data Type                   | Credit Scoring Application                                                                                               |
|----------------|-----------------------------|-------------------------------------------------------------------------------------------------------------------------|
| Traditional    | Bank transactional data     | Records of late payments on current and past credit, current loan amounts and loan purpose, credit history               |
| Traditional    | Credit bureau checks        | Number of credit inquiries                                                                                              |
| Traditional    | Commercial data             | Financial statements, number of working capital loans, and others                                                       |
| Alternative    | Utilities data              | Steady records of on-time payments as a possible indicator of creditworthiness                                          |
| Alternative    | Social media                | Social media data providing insights on consumer’s lifestyle                                                            |
| Alternative    | Credit scoring application  | Mobile applications                                                                                                     |
| Alternative    | Online transactions         | Data from online transactions                                                                                           |
| Alternative    | Behavioral data             | Mobile payment systems providing insights into the consumer’s behavior                                                  |
| Alternative    | Granular transactional data | Detailed insights into spending patterns                                                                                |
| Alternative    | Psychometrics               | Form filling data                                                                                                       |

## Modern Credit Scoring Systems
Modern credit scoring systems can collate data from a wide variety of sources in structured, unstructured, and semi-structured forms. New data sources include granular spending behavior, mobile data, geolocation data, and payment data from utilities and, in some cases, social media.

### Mobile Data
Mobile applications may collect data such as transport movements, geolocation, and transactional data. This data allows mobile phone applications to perform credit checks that traditional credit scoring providers (CSPs) may find challenging. However, data subjects may be unaware that their personal data is used for credit scoring.

### Social Media Data
Research suggests that the number and frequency of social media posts can provide insights into consumers' lifestyles, expenditures, and willingness to repay debt.

## Credit Scoring Methodologies

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

## Machine Learning in Credit Scoring

### Techniques Related to Credit Scoring Methods

#### Feature Engineering
Feature engineering involves transforming the input data set into meaningful features that help to reveal the underlying patterns in the data. This process enhances the accuracy of machine learning models by providing them with more relevant and informative data.

#### Reinforcement Learning
Reinforcement learning enables machines to learn behaviors based on feedback from the environment. This type of learning allows the model to adapt its actions to achieve optimal outcomes, either through a one-time learning process or continuous adaptation over time.

### Development of Machine Learning Models for Default Prediction
Developing a machine learning model for default prediction involves three phases:

1. **Preparation Phase:** Selects suitable machine learning algorithms and explores alternative data for credit scoring.
2. **Training Phase:** Trains the model using the selected algorithms and pre-processed data.
3. **Evaluation Phase:** Assesses the model's results, considers the outputs, and ensures model interpretability.

### Model Selection
Model selection is an exploratory process involving the continuous evaluation of multiple machine learning models. Some commonly used algorithms include Logistic Regression, Random Forest, Extra-Trees, CatBoost, LightGBM, XGBoost, K-Nearest Neighbours, Convolutional Neural Networks, and Stacking. (Refer to Appendix B for detailed descriptions of these algorithms.)

### Data Exploration
Credit scoring modeling requires rigorous exploratory data analysis (EDA) to distinguish variables and optimize the model.

### Model Training
Machine learning algorithms establish statistical/mathematical models that can make inferences. The training data (predictors/independent variables) and the outputs (responses/dependent variables) can be quantitative (numerical) or qualitative (categorical). Numerical outputs correspond to regression problems, while categorical outputs correspond to classification problems.

### Model Assessment
In the banking industry, model quality resides in the output and its interpretation. The model must be accurate to avoid losses and align with the bank's working capacity and liquidity. The Area Under the Receiver Operating Characteristic (ROC) Curve is a commonly used metric to assess model quality.

### Feature Importance
Investigating feature importance helps to further improve the model. Once the data has been explored and the model trained and tested, analyzing feature importance involves inspecting variables and assessing whether changes in a variable would affect the model output.

### Model Interpretability
Interpretability is the degree to which a human can understand the cause of a decision. Greater interpretability makes it easier for humans to understand why a decision or prediction was made. Model interpretation technologies include SHAP, ELI, LIME, Microsoft InterpretML, XAI, Alibi, TreeInterpreter, Skater, FairML, and fairness.


### Apps That Use Credit Scoring in Ethiopia

1. **Michu**
   - Michu is Ethiopia's first uncollateralized digital lending product, powered by Kifiya's Qena platform. It is designed for micro, small, and medium enterprises (MSMEs) and offers loans based on credit scoring.

2. **Telebirr Mela**
   - Telebirr Mela is a microloan service by Telebirr in partnership with Dashen Bank. It allows users to take out small loans without collateral based on their credit score.

3. **Alegnta**
   - Offered by Lion International Bank, Alegnta provides microloans and has a variety of loan products, including salary loans, loans for ride and taxi drivers, and loans for small and medium enterprises.

4. **Dube Ale**
   - Dube Ale is a collaboration between Dashen Bank and EagleLion System Technology. This app offers a Buy-Now Pay-Later scheme, enabling users to purchase goods and services on credit.

## Conclusion

Credit scoring is a fundamental component of modern financial systems, influencing a wide range of lending and credit decisions. By utilizing a variety of data sources and advanced methodologies, credit scoring models provide accurate assessments of creditworthiness, helping to mitigate risks for lenders and offering fair and transparent evaluations for borrowers. The integration of traditional techniques with innovative machine learning approaches enhances the predictive power and adaptability of these models.

In Ethiopia, the adoption of digital lending platforms and apps that leverage credit scoring has significantly improved financial inclusion. These platforms provide crucial access to credit for individuals and businesses, fostering economic growth and development. As credit scoring methodologies continue to evolve, they hold the potential to offer even more inclusive and effective financial solutions, benefiting a broader segment of society.

Through rigorous data analysis, model training, and continuous evaluation, credit scoring remains a dynamic and vital field, driving the future of lending and financial services.
