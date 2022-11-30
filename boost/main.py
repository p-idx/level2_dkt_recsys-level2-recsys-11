from args import parse_args
from dataloader import get_data, data_split
from models import get_model
from utils import setSeeds, transform_proba, save_prediction
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
    if args.wandb:
        wandb.init(entity='mkdir', project='kdg_cat_test', name=f'{args.model}_{args.fe_num}_{args.time_info}')
        wandb.config.update(args)

    print('------------------------load data------------------------')
    cate_cols, train_data, test_data = get_data(args)


    print('check by cv in catboost:',args.cat_cv)
    if args.cat_cv:
        cv_dataset = ctb.Pool(
                data=train_data.drop('answerCode', axis=1),
                label=train_data['answerCode'],
                cat_features=['userID']+cate_cols
                )
        
        params = {
          "iterations": args.n_epochs,
          "depth": args.depth,
          "loss_function": "Logloss",
        #   "custom_metric"='AUC',
          "verbose": args.verbose,
          "learning_rate": args.lr,
          "roc_file": "roc-file",
          "eval_metric": "AUC",
        #   "eval_set": eval_dataset
          }
        print('------------------------train model------------------------')
        _, model_list = ctb.cv(cv_dataset,
               params,
               fold_count=5,
               return_models=True
               )
        outputs = []
        print('------------------------predict------------------------')
        for model in model_list:
            pred = model.predict(test_data, prediction_type='Probability')
            
            output = transform_proba(pred)
                
            outputs.append(output)
            
        # Simple average ensemble
        predicts = np.mean(outputs, axis=0)



        
        print('------------------------save prediction------------------------')
        save_prediction(predicts, args)

        if args.wandb:
            print('log to wandb')
            out = pd.read_csv('./catboost_info/test_error.tsv', delimiter ='\t')
            wandb.define_metric("epochs")
            wandb.define_metric("metric", step_metric="epochs")

            for i in out.iter:
                epoch, metric, _ = out.loc[i]
                log_dict = {
                "epochs": epoch,
                "metric": metric,
                }
                wandb.log(log_dict)

    else:
        
        
        # train_data = pd.read_csv('/opt/ml/data/FE08/exp_train_data.csv')
        # valid_data = pd.read_csv('/opt/ml/data/FE08/exp_valid_data.csv')
        
        # X_train = train_data.drop('answerCode', axis=1)
        # y_train = train_data['answerCode']
        # X_valid = valid_data.drop('answerCode', axis=1)
        # y_valid = valid_data['answerCode']
        
        # print(X_train.shape, X_valid.shape)
        X_train, X_valid, y_train, y_valid = data_split(train_data, args.ratio)

        model = get_model(args)
        if args.model == 'CATB':
            model.fit(X_train, y_train,
                eval_set=(X_valid, y_valid),
                cat_features=['userID'] + cate_cols,
                early_stopping_rounds= 10,
                use_best_model=True,
                )
        elif args.model == 'LGB':
            model.fit(X_train, y_train,
                eval_set=(X_valid, y_valid),
                early_stopping_rounds= 10,
                )
        
        predicts = model.predict_proba(test_data)
        print(predicts.shape)
        output = []
        for zero, one in predicts:
            output.append(one)
        predicts = output

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

    if args.wandb:
        print('log to wandb')
        out = pd.read_csv('./catboost_info/test_error.tsv', delimiter ='\t')
        wandb.define_metric("epochs")
        wandb.define_metric("metric", step_metric="epochs")

        for i in out.iter:
            epoch, metric, _ = out.loc[i]
            log_dict = {
            "epochs": epoch,
            "metric": metric,
            }
            wandb.log(log_dict)


if __name__ == '__main__':

    args = parse_args()

    main(args)
