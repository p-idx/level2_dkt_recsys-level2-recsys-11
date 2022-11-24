import argparse
from dataloader import get_data, data_split
from models import get_model
from utils import setSeeds


# import hydra
# from omegaconf import DictConfig

# https://towardsdatascience.com/complete-tutorial-on-how-to-use-hydra-in-machine-learning-projects-1c00efcc5b9b


def main(args):

    setSeeds(args.seed)

    train_data, test_data = get_data(args)
    X_train, X_valid, y_train, y_valid = data_split(train_data)

    model = get_model(cfg)
    predicts = model(test_data)

    # SAVE

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
    parser.add_argument("--model", default="cat", type=str, help="model type")
    parser.add_argument("--n_epochs", default=100, type=int, help="number of epochs")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")

    args = parser.parse_args()

    main(args)
