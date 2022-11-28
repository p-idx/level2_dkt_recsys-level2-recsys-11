import argparse
from dataloader import get_data, data_split
from models import get_model
from utils import setSeeds
import os
import datetime
import wandb
import pandas as pd

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



    cate_cols, train_data, test_data = get_data(args)
    X_train, X_valid, y_train, y_valid = data_split(train_data, args.ratio)

    if args.model == 'LGB':
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_test = lgb.Dataset(X_valid, y_valid)

    if args.model == 'CATB':
        model = get_model(args)
        model.fit(X_train, y_train,
            eval_set=(X_valid, y_valid),
            cat_features=['userID'] + cate_cols,
            early_stopping_rounds= 10,
            )
    elif args.model == 'LGB':
        param = {'objective': 'binary',
                'metric': 'auc',
                'boosting': 'dart',
                'learning_rate': 0.03,
                'max_depth': 64,
                'num_leaves': 63
                }
        model = lgb.train(
            param,
            lgb_train,
            valid_sets=[lgb_train, lgb_test],
            verbose_eval=100,
            num_boost_round=2000,
            early_stopping_rounds=150,
        )



    predicts = model.predict(test_data)

    # SAVE
    output_dir = './output/'
    write_path = os.path.join(output_dir, f"{args.model}_{args.fe_num}_{args.time_info}.csv")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(predicts):
            w.write('{},{}\n'.format(id,p))

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

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default="/opt/ml/data/",
        type=str,
        help="data directory",
    )
    parser.add_argument(
        "--fe_num",
        default='00',
        type=str,
        help='feature engineering data file path (ex) 00'
    )
    parser.add_argument("--model", default="CATB", type=str, help="model type: CATB, LGB, XGB")
    parser.add_argument("--n_epochs", default=1000, type=int, help="number of epochs")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--ratio", default=0.3, type=float, help="test ratio")
    parser.add_argument("--wandb", default=False, type=bool, help="use wandb")

    parser.add_argument("--depth", default=6, type=int, help="depth of catboost")

    args = parser.parse_args()

    main(args)
