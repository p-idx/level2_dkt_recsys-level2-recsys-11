from args import parse_args
from dataloader import get_data, data_split
from models import get_model
from utils import setSeeds
import catboost as ctb
import os
import datetime
import wandb
import pandas as pd


# import hydra
# from omegaconf import DictConfig

# https://towardsdatascience.com/complete-tutorial-on-how-to-use-hydra-in-machine-learning-projects-1c00efcc5b9b


def main(args):
    args.time_info = (datetime.datetime.today() + datetime.timedelta(hours=9)).strftime('%m%d_%H%M')
    wandb.init(entity='mkdir', project='boost', name=f'{args.model}_{args.fe_num}_{args.time_info}')
    setSeeds(args.seed)
    
    wandb.config.update(args)
    
    cate_cols, train_data, test_data = get_data(args)
    print(train_data)
    
    if args.cat_cv:
        cv_dataset = ctb.Pool(data=train_data.drop('answerCode', axis=1),
                 label=train_data['answerCode'],
                 cat_features=cate_cols
                 )
        
        params = {"iterations": 300,
          "depth": 5,
          "loss_function": "Logloss",
          "verbose": True}

    X_train, X_valid, y_train, y_valid = data_split(train_data)
    
    model = get_model(args)
    model.fit(X_train, y_train,
            eval_set=(X_valid, y_valid),
            cat_features=cate_cols,
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
