import os
import datetime

from args import parse_args
from src.dataloader import DKTDataset, load_data
from src.utils import setSeeds
from src.model import LSTM
from src.lightning_model import DKTLightning

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold
import wandb


def main(args):
    wandb.login()

    hyperparameter_defaults = dict(
        model = args.model,
        fe_num = args.fe_num,
        max_seq_len = args.max_seq_len,
        cate_emb_dim = args.cate_emb_dim,
        cate_proj_dim = args.cate_proj_dim,
        cont_proj_dim = args.cont_proj_dim,
        hidden_dim = args.hidden_dim,
        n_layers = args.n_layers,
        n_heads = args.n_heads,
        drop_out = args.drop_out,
        batch_size = args.batch_size,
        learning_rate = args.lr,
    )
    
    setSeeds(args.seed)
    # args.device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data = load_data(args, is_train=True)

    # train_data ready
    train_data, valid_data = train_test_split(
        train_data, 
        test_size=0.3, # 일단 이 정도로 학습해서 추이 확인
        shuffle=True,
        random_state=args.seed,
    )

    train_dataset = DKTDataset(train_data, args)
    valid_dataset = DKTDataset(valid_data, args)

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

    # test_data ready
    test_data = load_data(args, is_train=False)
    test_dataset = DKTDataset(test_data, args)
    test_loader = DataLoader(
        test_dataset,
        num_workers=args.num_workers,
        shuffle=True,
        batch_size=args.batch_size,
    )

    test_loader = DataLoader(
        test_dataset,
        num_workers=4,
        shuffle=False,
        batch_size=args.batch_size,
    )

    # torch model, lightning model ready
    if args.model == 'LSTM':
        torch_model = LSTM(args)

    lightning_model = DKTLightning(args, torch_model)

    # about log, save model etc..
    args.time_info = (datetime.datetime.today() + datetime.timedelta(hours=9)).strftime('%m%d_%H%M')

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    write_path = os.path.join(
        args.model_dir,
        f"{torch_model.__class__.__name__}_{args.fe_num}_{args.time_info}/"
    )

    wandb_logger = WandbLogger( # 애가 wandb.init 비슷한거 다 해줌.
        entity='mkdir',
        project='yang3',
        name=f"{torch_model.__class__.__name__}_{args.fe_num}_{args.time_info}",
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
                monitor="valid_acc",
                filename="{epoch}-{valid_auc:.2f}-{valid_acc:.2f}",
                mode="max",
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


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
