import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import tqdm
import pdb
from scipy.sparse import csr_matrix, linalg
import warnings

import pandas as pd
import numpy as np
import catboost as ctb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
#import lightgbm as lgb

import random
import os
import re

import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler

from sklearn.model_selection import train_test_split
import os
import datetime
import wandb

from args import parse_args
from dataloader import get_data, data_split, option1_train_test_split, option1_5fold_train_test_split
from models import get_model
from utils import setSeeds, transform_proba, save_prediction, log_wandb
from sklearn.metrics import accuracy_score, roc_auc_score


warnings.filterwarnings(action='ignore')

def main(args):
    args.time_info = (datetime.datetime.today() + datetime.timedelta(hours=9)).strftime('%m%d_%H%M')
    setSeeds(args.seed)

    print('------------------------load data------------------------')
        
    cat_models={}
    def objective(trial):
        cate_cols, train_data, test_data, sub_test_data = get_data(args)
        X_train, X_valid, y_train, y_valid = option1_train_test_split(train_data, args)

        param = {
            "random_state":42,
            "objective" : "Logloss",
            'eval_metric' : 'AUC',
            "cat_features" : ['userID'] + cate_cols,
            'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.1),
            'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
            "n_estimators":trial.suggest_int("n_estimators", 1000, 10000),
            "max_depth":trial.suggest_int("max_depth", 4, 12),
            'random_strength' :trial.suggest_int('random_strength', 0, 100),
            "l2_leaf_reg":trial.suggest_float("l2_leaf_reg",0.1, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "max_bin": trial.suggest_int("max_bin", 200, 1000),
            'od_type': trial.suggest_categorical('od_type', ['IncToDec']),
            'od_pval': trial.suggest_float('od_pval', 0.01, 0.05),
            'od_wait': trial.suggest_int('od_wait', 50, 100),
        }
        # tmp = ctb.CatBoostClassifier()    
        model = ctb.CatBoostClassifier(**param, task_type = 'GPU')

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=100,
            # early_stopping_rounds=50
        )
        sub_y = sub_test_data['answerCode']
        sub_test_data = sub_test_data.drop(['answerCode'], axis=1)
        sub_preds = model.predict_proba(sub_test_data)[:, 1]
        sub_acc = accuracy_score(sub_y, np.where(sub_preds >= 0.5, 1, 0))
        sub_auc = roc_auc_score(sub_y, sub_preds)
        print(f"VALID AUC : {sub_auc} ACC : {sub_acc}\n")
        
        return sub_auc
        
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        study_name = 'cat_parameter_opt',
        direction = 'maximize',
        sampler = sampler,
    )
    study.optimize(objective, n_trials=50)
    print('=='*50)
    print(study.best_params)
    print('=='*50)
    ##### best_params로 최종 학습 (kfold) #####
    cate_cols, train_data, test_data, sub_test_data = get_data(args)
    sub_y = sub_test_data['answerCode']
    sub_test_data = sub_test_data.drop(['answerCode'], axis=1)
    model = ctb.CatBoostClassifier(**study.best_params, task_type='GPU', random_state=args.seed, objective='Logloss',
                                 cat_features=['userID'] + cate_cols)

    ### 5 fold start ###
    user_ids = option1_5fold_train_test_split(train_data)
    outputs = []
    for i,user_id in enumerate(user_ids):
        print('=='*20,f'fold {i+1} fitting', '=='*20)
        train = train_data[train_data["userID"].isin(user_id) == False]
        valid = train_data[train_data["userID"].isin(user_id)]

        # test데이터셋은 각 유저의 마지막 interaction만 추출
        # test = test[test["userID"] != test["userID"].shift(-1)]
        X_train = train.drop('answerCode', axis=1)
        X_valid = valid.drop('answerCode', axis=1)
        y_train = train['answerCode']
        y_valid = valid['answerCode']

        # 여기서 모델을 다시 선언해야 하나?
        model = ctb.CatBoostClassifier(**study.best_params, task_type='GPU', random_state=args.seed, objective='Logloss',
                                    cat_features=['userID'] + cate_cols)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=50)

        # sub inference (성능 확인차)
        sub_preds = model.predict_proba(sub_test_data)[:, 1]
        sub_acc = accuracy_score(sub_y, np.where(sub_preds >= 0.5, 1, 0))
        sub_auc = roc_auc_score(sub_y, sub_preds)
        print(f"VALID AUC : {sub_auc} ACC : {sub_acc}\n")

        # real inference
        fold_predicts = model.predict_proba(test_data)
        fold_predicts = transform_proba(fold_predicts)
        outputs.append(fold_predicts)
        save_prediction(fold_predicts, args, k=i+1, fold=True)

    predicts = np.mean(outputs, axis=0)

    # 5fold predictions SAVE
    save_prediction(predicts, args, M=True)

if __name__ == '__main__':

    args = parse_args()

    main(args)


print('optuna import complete')
