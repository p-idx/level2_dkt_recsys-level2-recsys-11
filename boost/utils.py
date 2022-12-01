import os
import random
import pandas as pd
import numpy as np
import torch
import wandb


def setSeeds(seed=42):

    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def transform_proba(pred:list):
    return list(zip(*pred))[1]
    output = []
    for zero, one in pred:
        output.append(one)
    return output


def save_prediction(predicts: list, args: dict, k=0):
    output_dir = './output/'
    if args.cat_cv:
        write_path = os.path.join(output_dir, f"{args.model}_{args.fe_num}_{args.time_info}_FOLD{k}.csv")
    else:
        write_path = os.path.join(output_dir, f"{args.model}_{args.fe_num}_{args.time_info}.csv")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(write_path, 'w', encoding='utf8') as w:
        print(f"writing prediction : {write_path}")
        w.write("id,prediction\n")
        for id, p in enumerate(predicts):
            w.write(f'{id},{p}\n')

def log_wandb(args):
    print('log to wandb')
    def read_error_file(valid_error, train_error):
        for i in valid_error.valid_iter:
            # TODO metric을 가변적으로 쓸 때도 wandb로 기록할 수 있게
            
            valid_metric = valid_error.loc[i].to_dict()
            valid_metric.update(train_error.loc[i].to_dict())
            del valid_metric['valid_iter']
            del valid_metric['train_iter']
            wandb.log(valid_metric)
            
    if args.wandb:
        if args.cat_cv:
            for k in range(args.FOLD_NUM):
                # args.k = k
                wandb.init(entity='mkdir', project='ksh_boost', name=f'{args.model}_{args.fe_num}_{args.time_info}_FOLD{k}')
                wandb.config.update(args)
                wandb.define_metric("iter")
                wandb.define_metric(f"{args.LOSS_FUNCTION}", step_metric="iter")
                
                valid_error = pd.read_csv(f'./catboost_info/fold-{k}/test_error.tsv', delimiter ='\t')
                train_error = pd.read_csv(f'./catboost_info/fold-{k}/learn_error.tsv', delimiter ='\t')
                
                read_error_file(valid_error, train_error)
                
                wandb.finish()
        else:
            wandb.init(entity='mkdir', project='ksh_boost', name=f'{args.model}_{args.fe_num}_{args.time_info}')
            wandb.config.update(args)
            wandb.define_metric("iter")
            wandb.define_metric(f"{args.LOSS_FUNCTION}", step_metric="iter")
            
            
            valid_error = pd.read_csv('./catboost_info/test_error.tsv', delimiter ='\t')
            train_error = pd.read_csv('./catboost_info/learn_error.tsv', delimiter ='\t')
            
            valid_column = valid_error.columns
            valid = []
            for col in valid_column:
                valid.append('valid_'+col)
            valid_error.columns = valid
            
            train_column = train_error.columns
            train = []
            for col in train_column:
                train.append('train_'+col)
            train_error.columns = train
            
            
            read_error_file(valid_error, train_error)