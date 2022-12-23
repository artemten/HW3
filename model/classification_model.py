from conf.conf import logging
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from util.util import save_model, load_model
from sklearn.model_selection import GridSearchCV
import numpy as np


def split(df):
    logging.info("Defining X and y")
    # variables
    X = df.iloc[:, :-1]
    y = df['target']
    logging.info("Splitting dataset")
    # Split variables into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, #independent variables
                                                        y, #dependent variable
                                                        random_state = 3
                                                    )
    return X_train, X_test, y_train, y_test

def gridsearch_random_forest(X_train, y_train):
    param_grid = {
    'max_depth': [80, 90, 100],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4],
    'min_samples_split': [8, 10],
    'n_estimators': [100, 200, 300]} 
    clf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    return grid_search.best_params_

def train_random_forest(X_train, y_train, best_params):
    # Initialize the model
    clf = RandomForestClassifier(random_state = 0)
    clf.set_params(**best_params)
    logging.info("Training the model")
    # Train the model
    clf.fit(X_train, y_train)
    save_model('model/conf/random_forest.pkl', clf)
    return clf

def gridsearch_logistic_regression(X_train, y_train):
    param_grid = {"C":np.logspace(-3,3,7), "penalty":["l2"]}
    clf = LogisticRegression(solver = 'liblinear', random_state = 0)
    grid_search = GridSearchCV(clf, param_grid,cv=3,scoring='precision')
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    return grid_search.best_params_

def train_logistic_regression(X_train, y_train, best_params):
    # Initialize the model
    clf = LogisticRegression(random_state = 3)
    clf.set_params(**best_params)
    logging.info("Training the model")
    # Train the model
    clf.fit(X_train, y_train)
    save_model('model/conf/logictic_regression.pkl', clf)
    return clf

def predict_data(values, path_to_model):
    clf = load_model(path_to_model)
    logging.info("Predicting values")
    # Predict values
    return clf.predict(values)