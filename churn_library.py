# library doc string


# import libraries
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

os.environ['QT_QPA_PLATFORM']='offscreen'



def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    return pd.read_csv(pth)
    


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    pass


def encoder_helper(df, category_lst, response="Churn"):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for category in category_lst:
        mean_encoded_col = df.groupby(category)[response].mean()
        df[f'{category}_{response}'] = df[category].map(mean_encoded_col)
    return df


def perform_feature_engineering(df, response="Churn"):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    if response not in df.columns:
        raise ValueError(f"{response} column not found in dataframe")
    
    y = df[response]
    X = pd.DataFrame()
    
    # List of categorical columns to encode
    category_lst = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    
    # Encoding categorical columns based on mean of response variable
    encoder_helper(df, category_lst, "Churn")
    #for col in categorical_cols:
    #    mean_encoded_col = df.groupby(col).mean()[response].to_dict()
    #    df[f'{col}_Churn'] = df[col].map(mean_encoded_col)
    
    # List of columns to keep
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'] + \
                [f'{col}_Churn' for col in category_lst]
    
    # Filtering the dataframe to keep only the necessary columns
    X = df[keep_cols]

    print(X.isnull().sum())
    
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test



def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass

class ModelHandler:
    """
    A class for handling model training, prediction, and saving to disk.
    """
    def __init__(self, model, param_grid=None, cv=None):
        """
        Initialize the ModelHandler.
        """
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        if self.param_grid:
            self.model = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv)
    
    def train(self, X, y):
        """
        Fit the model or GridSearchCV object.
        """
        self.model.fit(X, y)
    
    def predict(self, X):
        """
        Predict using the fitted model or the best estimator from GridSearchCV.
        """
        if isinstance(self.model, GridSearchCV):
            return self.model.best_estimator_.predict(X)
        else:
            return self.model.predict(X)
    
    def save_model(self, path):
        """
        Save the trained model to disk.
        """
        if isinstance(self.model, GridSearchCV):
            joblib.dump(self.model.best_estimator_, path)
        else:
            joblib.dump(self.model, path)

class RandomForestHandler(ModelHandler):
    """
    A class for handling RandomForestClassifier specific functionalities.
    """
    def __init__(self, random_state=42, param_grid=None, cv=None):
        super().__init__(RandomForestClassifier(random_state), param_grid, cv)

class LogisticRegressionHandler(ModelHandler):
    """
    A class for handling LogisticRegression specific functionalities.
    """
    def __init__(self, solver='lbfgs', max_iter=3000, param_grid=None, cv=None):
        super().__init__(LogisticRegression(solver=solver, max_iter=max_iter), param_grid, cv)


def main():
    df = import_data(r"./data/bank_data.csv")
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    print("isNull", df.isnull().sum())
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, "Churn")
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    ### TRAIN ###
    #rfc_param_grid = { 
    #    'n_estimators': [200, 500],
    #    'max_features': ['auto', 'sqrt'],
    #    'max_depth': [4, 5, 100],
    #    'criterion': ['gini', 'entropy']
    #}
    rfc_param_grid = {'n_estimators': [200, 500]}

    #rfc_handler = RandomForestHandler(param_grid=rfc_param_grid, cv=5)
    lrc_handler = LogisticRegressionHandler()
    
    # Assume X_train, y_train, X_test are defined elsewhere
    #rfc_handler.train(X_train, y_train)
    lrc_handler.train(X_train, y_train)
    
    #y_train_preds_rf = rfc_handler.predict(X_train)
    #y_test_preds_rf = rfc_handler.predict(X_test)
    
    y_train_preds_lr = lrc_handler.predict(X_train)
    y_test_preds_lr = lrc_handler.predict(X_test)
    
    #print("rf", y_train_preds_rf, y_test_preds_rf)
    print("lr", y_train_preds_lr, y_test_preds_lr)
    # Save models to disk
    #rfc_handler.save_model('rfc_model.joblib')
    #lrc_handler.save_model('lrc_model.joblib')

if __name__ == "__main__":
    main()