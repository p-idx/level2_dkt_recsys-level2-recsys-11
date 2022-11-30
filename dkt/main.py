import os
import datetime

from args import parse_args
from src.dataloader import DKTDataset, load_data
from src.utils import setSeeds
from src.model import LSTM, GRU, GRUBI, GRUATT, BERT, S2SGRU
from src.lightning_model import DKTLightning

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import wandb


def main(args):
    wandb.login()

    setSeeds(args.seed)

    # about log, save model etc..
    args.time_info = (datetime.datetime.today() + datetime.timedelta(hours=9)).strftime('%m%d_%H%M')

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # test_data, test_dataset, test_dataloader ready
    test_data = load_data(args, is_train=False)
    test_dataset = DKTDataset(test_data, args)
    test_loader = DataLoader(
        test_dataset,
        num_workers=args.num_workers,
        shuffle=False,
        batch_size=args.batch_size,
    )

    # train_data ready
    train_data = load_data(args, is_train=True)
    for_stratify = []
    for user_info in train_data:
        for col in user_info:
            if col.name == 'answerCode':
                for_stratify.append(col.iloc[-1])


    kf = StratifiedKFold(n_splits=5, shuffle=False)
    total_preds = np.zeros(len(test_data), dtype=np.float32)
    for i, (train_index, valid_index) in enumerate(kf.split(train_data, for_stratify)):
        if i != 4:
            continue
        # train_data_fold ready
        train_data_fold = train_data.iloc[train_index]
        valid_data_fold = train_data.iloc[valid_index]

        # train_data_fold dataset ready
        train_dataset = DKTDataset(train_data_fold, args)
        valid_dataset = DKTDataset(valid_data_fold, args)

        # train_data_fold dataloader ready
        train_loader = DataLoader(
            train_dataset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
        )
        valid_loader = DataLoader(
            valid_dataset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
        )
        args.k_i = i + 1   

        # torch model, lightning model ready
        if args.model == 'LSTM':
            torch_model = LSTM(args)
        elif args.model == 'GRU':
            torch_model = GRU(args)
        elif args.model == 'GRUBI':
            torch_model = GRUBI(args)
        elif args.model == 'GRUATT':
            torch_model = GRUATT(args)
        elif args.model == 'BERT':
            torch_model = BERT(args)
        elif args.model == 'S2SGRU':
            torch_model = S2SGRU(args)

        lightning_model = DKTLightning(args, torch_model.to('cuda'))

        write_path = os.path.join(
            args.model_dir,
            f"{args.model}_{args.time_info}_K{args.k_i}_{args.leak}_FE{args.fe_num}/"
        )

        wandb_logger = WandbLogger(
            entity='mkdir',
            project='yang',
            name=f"{args.model}_{args.fe_num}_{args.time_info}_K{args.k_i}_{args.leak}_FE{args.fe_num}",
        )

        wandb_logger.experiment.config.update(args)

        # trainer ready
        trainer = pl.Trainer(
            default_root_dir=os.getcwd(), 
            logger=wandb_logger,
            log_every_n_steps=args.log_steps,
            callbacks=[
                EarlyStopping(
                    monitor='valid_loss', 
                    mode='min', 
                    patience=args.patience,
                    verbose=True,
                ),
                ModelCheckpoint(
                    dirpath=write_path,
                    monitor="valid_loss",
                    filename=os.path.join(write_path, "valid_loss_min"),
                    mode="min",
                    save_top_k=1,
                ),
            ],
            gradient_clip_val=args.clip_grad,
            max_epochs=args.n_epochs,
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
        args.output_dir, 
        f"{args.model}_{args.time_info}_M_{args.leak}_FE{args.fe_num}.csv"
    )

    total_preds /= 5
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write("{},{}\n".format(id, p))

            
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
