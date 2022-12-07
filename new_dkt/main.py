import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold

import os
import datetime

from src.dataloader import get_data, get_loader
from src.utils import setSeeds
from src.models import *
# from src.trainer import run
from src.trainer import DKTLightning
import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import argparse


def main(config):
    wandb.login()

    X_train, y_train = get_data(config, is_train=True)
    X_test, y_test = get_data(config, is_train=False)
    test_loader = get_loader(config, X_test, y_test, shuffle=False)

    
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)


    config.time_info = (datetime.datetime.today() + datetime.timedelta(hours=9)).strftime('%m%d_%H%M')
    

    for_stratify = []
    for answer in y_train:
        if answer[0].values[-1] == 1:
            for_stratify.append(1)
        else:
            for_stratify.append(0)


    kf = StratifiedKFold(n_splits=5, shuffle=False)
    total_preds = np.zeros(len(y_test), dtype=np.float32)
    for i, (train_index, valid_index) in enumerate(kf.split(X_train, for_stratify)):
        if i != 0:
            break

        X_train_fold = X_train.iloc[train_index]
        X_valid_fold = X_train.iloc[valid_index]

        y_train_fold = y_train.iloc[train_index]
        y_valid_fold = y_train.iloc[valid_index]


        train_loader = get_loader(config, X_train_fold, y_train_fold, shuffle=True)
        valid_loader = get_loader(config, X_valid_fold, y_valid_fold, shuffle=True)
        config.k_i = i + 1   

        # torch model, lightning model ready
        if config.model == 'LSTM':
            model = LSTM(config)
        elif config.model == 'SAKT':
            model = SAKT(config)
        elif config.model == 'LastQuery':
            model = LastQuery(config)

        lightning_model = DKTLightning(config, model.to(config.device))

        write_path = os.path.join(
            'models/',
            f"{config.model}_{config.time_info}_K{config.k_i}_FE{config.cate_cols + config.cont_cols}/"
        )

        wandb_logger = WandbLogger(
            entity='mkdir',
            project='new_yang2',
            name=f"{config.model}_{config.time_info}_K{config.k_i}_FE{config.cate_cols + config.cont_cols}",
        )

        wandb_logger.experiment.config.update(config)

        # trainer ready
        trainer = pl.Trainer(
            default_root_dir=os.getcwd(), 
            logger=wandb_logger,
            log_every_n_steps=10,
            callbacks=[
                EarlyStopping(
                    monitor='valid_auc', 
                    mode='max', 
                    patience=5,
                    verbose=True,
                ),
                ModelCheckpoint(
                    dirpath=write_path,
                    monitor="valid_auc",
                    filename=os.path.join(write_path, "valid_auc_max"),
                    mode="max",
                    save_top_k=1,
                ),
            ],
            gradient_clip_val=config.clip_grad,
            max_epochs=config.epochs,
            accelerator='gpu'
        )

        # train
        trainer.fit(lightning_model, train_loader, valid_loader)

        # inference
        preds = trainer.predict(lightning_model, test_loader)
        total_preds += torch.concat(preds).numpy()
        wandb.finish()
        # break
        
        
    # kfold mean ensemble
    write_path = os.path.join(
        config.output_dir, 
        f"{config.model}_{config.time_info}_M_FE{config.cate_cols + config.cont_cols}.csv"
    )

    total_preds /= 5
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write("{},{}\n".format(id, p))


if __name__ == '__main__':
    setSeeds()

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default="cuda", type=str, help="cpu or gpu")
    parser.add_argument("--model_dir", default="models/", type=str, help="model directory")
    parser.add_argument("--output_dir", default="output/", type=str, help="output directory")


    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--learning_rate", default=0.0001, type=float)
    parser.add_argument("--drop_out", default=0.4, type=float)
    parser.add_argument("--clip_grad", default=0.75, type=str)
    parser.add_argument("--loss", default='bce', type=str)
    parser.add_argument("--model", default='LastQuery', type=str)


    parser.add_argument("--inter_embed_size", default=16, type=int)
    parser.add_argument("--cate_embed_size", default=32, type=int)
    parser.add_argument("--cont_embed_size", default=2, type=int) # 콘티 컬럼 개수에 따라 같이 조정 필요, 거의 상관은 없음


    parser.add_argument("--seq_len", default=64, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--num_layers", default=1, type=int)


    parser.add_argument("--attention_size", default=32, type=str)
    parser.add_argument("--ffn_size", default=256, type=str)
    parser.add_argument("--num_heads", default=2, type=int)

    # 변수 관련
    parser.add_argument("--cate_cols", 
        default=['assessmentItemID', 'testId', 'KnowledgeTag', 'testId_large', 'testId_avg_rate', 'assessmentItemID_avg_rate'], 
        nargs='+', 
        type=str
    )
    # 타임스탬프를 drop 하니, 타임스탬프 관련 좋은 EDA 없을까?, 나중에 GCN 성공한다면 그 레이턴트 벡터도?
    parser.add_argument("--cont_cols", 
        default=['user_elapse', 'know_tag_emb0', 'know_tag_emb1'], 
        nargs='+', 
        type=str
    )

    config = parser.parse_args()

    main(config)
