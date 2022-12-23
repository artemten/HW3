#code
from model.classification_model import predict_data, split, train_random_forest, train_logistic_regression, gridsearch_random_forest, gridsearch_logistic_regression
from conf.conf import logging
from connector.connector import get_data
from conf.conf import settings
from util.util import load_model


settings.load_file(path="conf/setting.toml")



df = get_data(settings.DATA.data_set)
X_train, X_test, y_train, y_test = split(df)
best_params = gridsearch_logistic_regression(X_train, y_train)
clf = train_logistic_regression(X_train, y_train, best_params)
logging.info(f'Accuracy is {clf.score(X_test, y_test)}')
clf = load_model(settings.MODEL.dt_conf)
logging.info(f'Prediction is {clf.predict(X_test)}')


logging.info(f"prediction: {predict_data(X_test, settings.MODEL.dt_conf)}")