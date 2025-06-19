import os
import sys
from pathlib import Path
from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable
import time
import json
import gc
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import optuna
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
    
seed = 42


def xgb_params(trial):
    return {
        # fixed
        'objective': trial.suggest_categorical('objective', ['reg:squarederror']),
        'tree_method': trial.suggest_categorical('tree_method', ['hist']),
        'device': trial.suggest_categorical('device', ['cuda']),
        # 'predictor': trial.suggest_categorical('predictor', ['gpu_predictor']),
        'random_state': trial.suggest_categorical('random_state', [seed]),
        # hyperparams
        'n_estimators': trial.suggest_int('n_estimators', 1000, 2000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'subsample': trial.suggest_float('subsample', 0.01, 0.25, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.7),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.1, 0.7),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.1, 0.7),
        'lambda': trial.suggest_float('lambda', 10, 200, log=True),
        'alpha': trial.suggest_float('alpha', 10, 100, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 3.0),
    }# other: n_jobs, random_state, verbose, max_leaves, min_child_weight



def lgbm_params(trial):
    return {
        # fixed
        'objective': trial.suggest_categorical('objective', ['regression']),
        'device': trial.suggest_categorical('device', ['gpu']),
        'random_state': trial.suggest_categorical('random_state', [seed]),
        'verbose': trial.suggest_categorical('verbose', [0]),  # no output
        # hyperparams
        'n_estimators': trial.suggest_int('n_estimators', 1000, 2000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'subsample': trial.suggest_float('subsample', 0.01, 0.25, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.7),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.3, 0.7),
        'reg_alpha': trial.suggest_float('reg_alpha', 10, 100, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 10, 200, log=True),
    }# other? goss



def catb_params(trial):
    return {
        # fixed
        # 'loss_function': trial.suggest_categorical('loss_function', ['RMSE']),
        'task_type': trial.suggest_categorical('task_type', ['GPU']),
        'random_state': trial.suggest_categorical('random_state', [seed]),
        'verbose': trial.suggest_categorical('verbose', [0]),  # no output
        'leaf_estimation_iterations': trial.suggest_categorical('leaf_estimation_iterations', [5]), #default 10
        # hyperparams
        'n_estimators': trial.suggest_int('n_estimators', 1000, 2000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'depth': trial.suggest_int('depth', 3, 7),
        'subsample': trial.suggest_float('subsample', 0.01, 0.25, log=True),
        #'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.05, 0.25, log=True), # only supported on cpu
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bernoulli']),  
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 10, 200, log=True),
    }# other? max_bin?




# ive decided to use naive nested 5 fold to do the tuning. 
# maybe setting one of them to None can yield back simple kfold. 
# not sure what to do yet for ensembling, but this is an issue for way later


def do_opuna_optimization(
    X: np.ndarray,
    y: np.ndarray,
    logs_dir: Path,
    ModelClass: BaseEstimator,
    params_fn: Callable = xgb_params,
    n_trials: int = 100,
    cv = KFold(n_splits=5, shuffle=False),

):
    #Configure logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    model_log_dir = logs_dir / ModelClass.__name__
    os.makedirs(model_log_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    logger.addHandler(logging.FileHandler(model_log_dir / f"optuna{timestamp}.log", mode="w"))  # Log to a file named "optuna.log"
    optuna.logging.enable_propagation()
    
    #optuna objective
    def objective(trial):
        params = params_fn(trial)
        scores = []
        for train_idx, valid_idx in cv.split(X, y):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]
            mdl = ModelClass(**params)
            mdl.fit(X_train, y_train)
            preds = mdl.predict(X_valid)
            rho, _ = pearsonr(y_valid, preds)
            scores.append(rho)
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", 
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    print("Best trial:", study.best_trial.number)
    print("Best value (CV RMSE):", study.best_value)
    print("Best params:", study.best_params)
    return study








######################################################  |
#####  command line argument to run experiments  #####  |
######################################################  V

# start by using all datasets. I can change this later
def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments with different models.")
    parser.add_argument(
        "--models", 
        nargs='+', 
        type=str, 
        default=["LightGBM", "XGBoost", "CatBoost"], 
        help="List of model names to run."
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="/home/nikita/Code/random-feature-boosting/save/OpenMLRegression/",
        help="Directory where the results json will be saved to file."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/nikita/Code/random-feature-boosting/save/OpenMLRegression/",
        help="Directory where the results json will be saved to file."
    )
    parser.add_argument(
        "--n_optuna_trials",
        type=int,
        default=100,
        help="Number of optuna trials in the inner CV hyperparameter loop."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for all randomness."
    )
    parser.add_argument(
        "--kfolds",
        type=int,
        default=5,
        help="Number of folds for optuna cv."
    )
    return parser.parse_args()



if __name__ == "__main__":
    #args
    args = parse_args()
    
    #load dataset
    train_X = pd.read_parquet(Path(args.data_dir) / "train.parquet").astype(np.float32)
    train_y = train_X.pop("label")

    # Run experiments
    for model_name in args.models:
        if model_name.lower() == "xgboost":
            ModelClass = XGBRegressor
            params_fn = xgb_params
        elif model_name.lower() == "lightgbm":
            ModelClass = LGBMRegressor
            params_fn = lgbm_params
        elif model_name.lower() == "catboost":
            ModelClass = CatBoostRegressor
            params_fn = catb_params
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # run the experiments
        do_opuna_optimization(
            X=train_X.values,
            y=train_y.values,
            logs_dir=Path(args.logs_dir),
            ModelClass=ModelClass,
            params_fn=params_fn,
            n_trials=args.n_optuna_trials,
            cv=KFold(n_splits=args.kfolds, shuffle=False),
        )