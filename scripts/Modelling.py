import logging , os , pandas as pd 
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score , roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Format of the log messages
)
# Create a logger object
logger = logging.getLogger(__name__)

# define the path to the Logs directory one level up
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','logs')

# create the logs directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# define file paths
log_file_info = os.path.join(log_dir, 'info.log')
log_file_error = os.path.join(log_dir, 'error.log')

# Create handlers
info_handler = logging.FileHandler(log_file_info)
info_handler.setLevel(logging.INFO)

error_handler = logging.FileHandler(log_file_error)
error_handler.setLevel(logging.ERROR)

# Create a formatter and set it for the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

# Create a logger and set its level
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Capture all info and above
logger.addHandler(info_handler)
logger.addHandler(error_handler)


def load_data(path):
    logger.info("importting the data ")
    try:
        # my data have 'Label' column for binary classification
        logger.info("loading the data ")
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"error occured while loading the data {e}")
def preprocess_data(data):
    logger.info("Preprocessing the data")
    try:
        # Drop irrelevant columns
        data = data.drop(columns=['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId'])
        # Handle missing values: Impute 'Std_Transaction_Amount' with the median
        data['Std_Transaction_Amount'].fillna(data['Std_Transaction_Amount'].median(), inplace=True)

        # Convert categorical columns to one-hot encoding if necessary
        categorical_columns = ['CurrencyCode', 'CountryCode', 'ProductId', 'ChannelId']  # Include 'ChannelId'
        data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

        # Handle 'TransactionStartTime' by extracting useful datetime features
        if 'TransactionStartTime' in data.columns:
            logger.info("Extracting datetime features from 'TransactionStartTime'")
            data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
            data['TransactionHour'] = data['TransactionStartTime'].dt.hour
            data['TransactionDay'] = data['TransactionStartTime'].dt.day
            data['TransactionMonth'] = data['TransactionStartTime'].dt.month
            data['TransactionWeekday'] = data['TransactionStartTime'].dt.weekday

            # Drop the original 'TransactionStartTime' column after extracting features
            data = data.drop(columns=['TransactionStartTime'])

        logger.info("Data preprocessing completed")
        return data
    except Exception as e:
        logger.error(f"Error occurred while preprocessing the data: {e}")
        return None


def split_the_data(data):
    logger.info("spliting the data ")
    try:
        X= data.drop(columns=['Label']) # Features
        y = data['Label'] # Tareget variable
        X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2 , random_state=42 , stratify=y)
        return X_train , X_test , y_train , y_test 
    except Exception as e:
        logger.info(f"error occured {e}")
# i use two models which are Logidtic Regression and Random Forst
def tain_the_models(X_train,y_train,X_test):
    logger.info("train the model")
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
       
        logger.info("Initializing the model")
        # Initialize the models
        logistic_model = LogisticRegression(max_iter=1000 , random_state=42)
        random_forest_model = RandomForestClassifier(random_state=42)
        logger.info("training the model with our data")
        # Train the models 
        logistic_model.fit(X_train_scaled,y_train)
        random_forest_model.fit(X_train,y_train)
        return logistic_model , random_forest_model
    except Exception as e:
        logger.error(f"error occured {e}")


def grid_search_tuning(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_
    return best_params, best_estimator

def random_search_tuning(model, param_distributions, X_train, y_train, n_iter=100):
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=n_iter, cv=5, scoring='accuracy', random_state=42)
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    best_estimator = random_search.best_estimator_
    return best_params, best_estimator


# model evaluation 
def evaluate_models(model,X_test,y_test):
    logger.info("Evaluate the models")
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)
        roc_auc = roc_auc_score(y_test , model.predict_proba(X_test)[:,1])

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")

        return y_pred
    except Exception as e:
        logger.error(f"error occures{e}")