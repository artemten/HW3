#code
from model.decision_tree import predict
from conf.conf import logging


df = get_data('https://raw.githubusercontent.com/5x12/ml-cookbook/master/supplements/data/heart.csv')
X_train, X_test, y_train, y_test = split(df)
clf = train_decision_tree(X_train, y_train)
logging.info(f'Accuracy is {clf.score(X_test, y_test)}')

response = predict(X_test)
clf = load_model('model/conf/decision_tree.pkl')
logging.info(f'Prediction is {clf.predict(X_test)}')