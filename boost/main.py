from args import parse_args
from dataloader import get_data, data_split
from models import get_model
from utils import setSeeds, transform_proba, save_prediction, log_wandb
import catboost as ctb
import os
import datetime
import wandb
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


import lightgbm as lgb

# import hydra
# from omegaconf import DictConfig

# https://towardsdatascience.com/complete-tutorial-on-how-to-use-hydra-in-machine-learning-projects-1c00efcc5b9b


def main(args):
    args.time_info = (datetime.datetime.today() + datetime.timedelta(hours=9)).strftime('%m%d_%H%M')
    setSeeds(args.seed)

    print('------------------------load data------------------------')
    cate_cols, train_data, test_data = get_data(args)

    print('check by cv in catboost:',args.cat_cv)
    if args.cat_cv:
        cv_dataset = ctb.Pool(
                data=train_data.drop('answerCode', axis=1),
                label=train_data['answerCode'],
                cat_features=['userID']+cate_cols
                )
        args.LOSS_FUNCTION = 'Logloss'
        params = {
          "iterations": args.n_epochs,
          "depth": args.depth,
          "loss_function": args.LOSS_FUNCTION,
        #   "custom_metric"='AUC',
          "verbose": args.verbose,
          "learning_rate": args.lr,
          "roc_file": "roc-file",
          "eval_metric": "AUC",
        #   "eval_set": eval_dataset
          }
        print('------------------------train model------------------------')
        args.FOLD_NUM = 2
        _, model_list = ctb.cv(cv_dataset,
               params,
               fold_count=args.FOLD_NUM,
               return_models=True
               )
        print('log to wandb')
        log_wandb(args)

        outputs = []
        print('------------------------predict------------------------')
        for fold, model in enumerate(model_list):
            pred = model.predict(test_data, prediction_type='Probability')
            output = transform_proba(pred)
            save_prediction(output, args, k=fold)
            outputs.append(output)
            
        # Simple average ensemble
        predicts = np.mean(outputs, axis=0)
        
        print('------------------------save prediction------------------------')
        save_prediction(predicts, args)

    else:
        X_train, X_valid, y_train, y_valid = data_split(train_data, test_data, args)
        
        
        model = get_model(args)
        if args.model == 'CATB':
            model.fit(X_train, y_train,
                eval_set=(X_valid, y_valid),
                cat_features=['userID'] + cate_cols,
                # early_stopping_rounds= 50,
                use_best_model=True,
                )
        elif args.model == 'LGB':
            model.fit(X_train, y_train,
                eval_set=(X_valid, y_valid),
                early_stopping_rounds= 50,
                )
        
        predicts = model.predict_proba(test_data)
        predicts = transform_proba(predicts)
        # print(predicts.shape)
        # output = []
        # for zero, one in predicts:
        #     output.append(one)
        # predicts = output

        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        print('피쳐 별 중요도')
        for i, j in zip(np.array(test_data.columns)[sorted_idx], feature_importance[sorted_idx]):
            print(f'{i:20}:{j}')
        fig = plt.figure(figsize=(12, 8))
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center', color='green')
        plt.yticks(range(len(sorted_idx)), np.array(test_data.columns)[sorted_idx])
        plt.title('Feature Importance')
        plt.show()
        plt.savefig('Test.pdf')

        # SAVE
        save_prediction(predicts, args)
        
        log_wandb(args)


if __name__ == '__main__':

    args = parse_args()

    main(args)
