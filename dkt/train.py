import os

import torch
import wandb

from args import parse_args
from src import trainer
from src.dataloader import Preprocess, get_data, split_data
from src.utils import setSeeds


def main(args):
    wandb.login()

    setSeeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # train_data = get_data(args, is_train=True)
    # raise 'ok'
    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()
    print(train_data)
    raise 'o'
    # data leakage?-> merge.
    
    # metric = basic, kfold
    # 모델 앞단에서까지 자동화 하고, train 좀 더 깔끔하게 바꾸자. 그 이후에 kfold 하자.

    train_data, valid_data = preprocess.split_data(train_data)

    wandb.init(project="dkt", config=vars(args))
    model = trainer.get_model(args).to(args.device)
    trainer.run(args, train_data, valid_data, model)




if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
