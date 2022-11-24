import argparse
from dataloader import get_data, data_split
from models import get_model
from utils import setSeeds
import os
import datetime


# import hydra
# from omegaconf import DictConfig

# https://towardsdatascience.com/complete-tutorial-on-how-to-use-hydra-in-machine-learning-projects-1c00efcc5b9b


def main(args):
    setSeeds(args.seed)
    args.time_info = (datetime.datetime.today() + datetime.timedelta(hours=9)).strftime('%m%d_%H%M')

    cate_cols, train_data, test_data = get_data(args)
    X_train, X_valid, y_train, y_valid = data_split(train_data)

    model = get_model(args)
    model.fit(X_train, y_train,
            eval_set=(X_valid, y_valid),
            cat_features=cate_cols)

    predicts = model.predict(test_data)

    # SAVE
    output_dir = './output/'
    write_path = os.path.join(output_dir, f"{args.model}_{args.time_info}.csv")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(predicts):
            w.write('{},{}\n'.format(id,p))


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
    parser.add_argument("--model", default="CATB", type=str, help="model type")
    parser.add_argument("--n_epochs", default=100, type=int, help="number of epochs")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--seed", default=42, type=int, help="seed")

    parser.add_argument("--depth", default=6, type=int, help="depth of catboost")

    args = parser.parse_args()

    main(args)
