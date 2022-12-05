import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser.add_argument(
        "--data_dir",
        default="/opt/ml/level2_dkt_recsys-level2-recsys-11/data/",
        type=str,
        help="data directory",
    )
    parser.add_argument(
        "--fe_num",
        default='00',
        type=str,
        help='feature engineering data file path (ex) 00'
    )
    
    parser.add_argument("--ratio", default=0.3, type=float, help="test ratio")
    parser.add_argument("--valid_exp", default='True', type=str2bool, help="use last n row for valid set")
    parser.add_argument("--valid_exp_n", default=1, type=int, help="how many use last row for valid")

    parser.add_argument("--model", default="CATB", type=str, help="model type")
    parser.add_argument("--cat_cv", default='False', type=str2bool, help="cross validation of catboost")
    parser.add_argument("--n_epochs", default=15000, type=int, help="number of epochs")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--verbose", default=50, type=int, help="number of verbose")
    parser.add_argument("--od_type", default='Iter', type=str, help="catboost's od_type")
    parser.add_argument("--LOSS_FUNCTION", default='Logloss', type=str, help="catboost's loss function")
    parser.add_argument("--depth", default=6, type=int, help="depth of catboost")
    parser.add_argument("--l2_leaf_reg", default=6, type=int, help="depth of catboost")
    parser.add_argument("--grow_policy", default='Lossgiude', type=str, help="tree grow policy")

    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--wandb", default='False', type=str2bool, help="use wandb")
    parser.add_argument("--is_new", default='False', type=str2bool, help="use new validation split process")


    args = parser.parse_args()

    return args