"""
This is the demo code that uses hydra to access the parameters in under the directory config.

Author: Khuyen Tran
"""

import hydra
from omegaconf import DictConfig

import warnings

warnings.filterwarnings(action="ignore")

from functools import partial
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
#from xgboost import XGBClassifier

def load_data(path: DictConfig):
    X_train = pd.read_csv(abspath(path.X_train.path))
    X_test = pd.read_csv(abspath(path.X_test.path))
    y_train = pd.read_csv(abspath(path.y_train.path))
    y_test = pd.read_csv(abspath(path.y_test.path))

    return X_train, X_test, y_train, y_test

# def get_objective(
#     X_train: pd.DataFrame,
#     y_train: pd.DataFrame,
#     X_test: pd.DataFrame,
#     y_test: pd.DataFrame,
#     config: DictConfig,
#     space: dict,
# ):
#     model = LogisticRegression(solver='lbfgs'
#    """  penalty='l2', 
#     dual=False, 
#     tol=0.0001, 
#     C=1.0, 
#     fit_intercept=True,
#     intercept_scaling=1, 
#     class_weight=None, 
#     random_state=None,
#     solver='lbfgs',
#     max_iter=100, 
#     multi_class='deprecated',
#     verbose=0, 
#     warm_start=False, 
#     n_jobs=None, 
#     l1_ratio=None) """



# """         use_label_encoder=config.model.use_label_encoder,
#         objective=config.model.objective,
#         n_estimators=space["n_estimators"],
#         max_depth=int(space["max_depth"]),
#         gamma=space["gamma"],
#         reg_alpha=int(space["reg_alpha"]),
#         min_child_weight=int(space["min_child_weight"]),
#         colsample_bytree=int(space["colsample_bytree"]),
#         eval_metric=config.model.eval_metric,
#         early_stopping_rounds=config.model.early_stopping_rounds, """
#     )

#     evaluation = [(X_train, y_train), (X_test, y_test)]

#     model.fit(
#         X_train,
#         y_train,
#         eval_set=evaluation,
#     )
#     prediction = model.predict(X_test.values)
#     accuracy = accuracy_score(y_test, prediction)
#     print("SCORE:", accuracy)
#     return {"loss": -accuracy, "status": STATUS_OK, "model": model}

# """ def optimize(objective: Callable, space: dict):
#     trials = Trials()
#     best_hyperparams = fmin(
#         fn=objective,
#         space=space,
#         algo=tpe.suggest,
#         max_evals=100,
#         trials=trials,
#     )
#     print("The best hyperparameters are : ", "\n")
#     print(best_hyperparams)
#     best_model = trials.results[
#         np.argmin([r["loss"] for r in trials.results])
#     ]["model"]
#     return best_model """

@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def train_model(config: DictConfig):
    """Function to train the model"""
    """ 
    X_train, X_test, y_train, y_test = load_data(config.processed)
    
    # Define space
    space = {
        "max_depth": hp.quniform("max_depth", **config.model.max_depth),
        "gamma": hp.uniform("gamma", **config.model.gamma),
        "reg_alpha": hp.quniform("reg_alpha", **config.model.reg_alpha),
        "reg_lambda": hp.uniform("reg_lambda", **config.model.reg_lambda),
        "colsample_bytree": hp.uniform(
            "colsample_bytree", **config.model.colsample_bytree
        ),
        "min_child_weight": hp.quniform(
            "min_child_weight", **config.model.min_child_weight
        ),
        "n_estimators": config.model.n_estimators,
        "seed": config.model.seed,
    }

    objective = partial(get_objective, X_train, y_train, X_test, y_test, config)

    # Encuentra el mejor modelo
    best_model = optimize(objective, space) """

    X_train, X_test, y_train, y_test = load_data(config.processed)
    # Initialize the models
    #knn = KNeighborsClassifier()
    log_reg = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
    #decision_tree = DecisionTreeClassifier()
    #svm = SVC()
    print(X_train)
    print("y_train")
    print(y_train)

    # Dictionary to store models and their respective accuracies
    models = {
    #    'K-Nearest Neighbors': knn,
        'Logistic Regression': log_reg,
     #   'Decision Tree': decision_tree,
      #  'Support Vector Machine': svm
    }

    # Dictionary to store accuracies
    accuracies = {}

    # Train and evaluate each model
    for model_name, model in models.items():
        # Train the model
        best_model= model.fit(X_train, y_train)
        print("train success")
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        print("pred success")
        # Calculate the accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[model_name] = accuracy
        print(accuracy)
    # Guarda el modelo
    joblib.dump(best_model, abspath(config.model.path))

    
    #y_pred2= model.predict()
    #print(f"Train modeling using {config.data.processed}")
    print(f"Model used: {config.model.name}")
    #print(f"Save the output to {config.data.final}")


if __name__ == "__main__":
    train_model()
