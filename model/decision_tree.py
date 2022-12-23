from connector.connector import get_data
from conf.conf import logging
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from util.util import save_model, load_model
from conf.conf import settings


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


def train_decision_tree(X_train, y_train):
    # Initialize the model
    clf = DecisionTreeClassifier(max_depth = 3,
                                random_state = 3
                                )
    logging.info("Training the model")
    # Train the model
    clf.fit(X_train, y_train)
    save_model('model/conf/decision_tree.pkl', clf)
    return clf


def predict(values, path_to_model):
    clf = load_model(path_to_model)
    return clf.predict(values)