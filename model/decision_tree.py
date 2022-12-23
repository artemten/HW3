from connector.connector import get_data
from conf.conf import logging

logging.INFO("extracting dataset")
df = get_data('https://raw.githubusercontent.com/5x12/ml-cookbook/master/supplements/data/heart.csv')

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def train_test_split_split(df):
    # variables
    X = df.iloc[:, :-1]
    y = df['target']
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
    # Train the model
    clf.fit(X_train, y_train)
    return clf