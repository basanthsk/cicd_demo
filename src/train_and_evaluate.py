'''
load the train and test files
train the with algorithm
save the matrics and parameters
'''


import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from urllib.parse import urlparse
from get_data import read_params, get_config
import joblib
import json
import pickle


def eval_metrics(actual, pred):
    tn, fp, fn, tp = confusion_matrix(actual, pred).ravel()
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    specificity = tn/(fp+tn)
    F1_Score = 2*(recall * precision) / (recall + precision)
    result = {"Accuracy": accuracy, "Precision": precision,
              "Recall": recall, 'Specficity': specificity, 'F1': F1_Score}
    return result


def train_and_eval(config_path):
    config = read_params(config_path)
    x_train_path = config["pre_proccess"]["x_train_path"]
    y_train_path = config["pre_proccess"]["y_train_path"]
    x_test_path = config["pre_proccess"]["x_test_path"]
    y_test_path = config["pre_proccess"]["y_test_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    # alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    # l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]

    solver = config["estimators"]["Logistic"]["params"]["solver"]
    verbose = config["estimators"]["Logistic"]["params"]["verbose"]

    target = [config["base"]["label"]]

    train_y = pd.read_csv(y_train_path, sep=",")
    test_y = pd.read_csv(y_test_path, sep=",")

    train_x = pd.read_csv(x_train_path, sep=",")
    test_x = pd.read_csv(x_test_path, sep=",")

    lr = LogisticRegression(verbose=verbose, solver=solver)
    # lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    lr.fit(train_x, train_y)
    webapp_path = config["webapp_model_dir"]

    pickle.dump(lr, open(webapp_path, "wb"))
    predicted_qualities = lr.predict(test_x)

    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(scores_file, "w") as f:
        scores = eval_metrics(test_y, predicted_qualities)
        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:
        params = {
            "solver": solver,
            "verbose": verbose
        }
        json.dump(params, f, indent=4)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(lr, model_path)


if __name__ == "__main__":
    train_and_eval(config_path=get_config())
