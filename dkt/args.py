import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")

    # parser.add_argument("--device", default="cpu", type=str, help="cpu or gpu")

    parser.add_argument('--userid', default=0, type=int, help='userID 학습 유무, 학습 시 concat 진행한 train 으로 학습.')

    parser.add_argument(
        "--data_dir",
        default="/opt/ml/level2_dkt_recsys-level2-recsys-11/data/",
        type=str,
        help="data directory",
    )

    parser.add_argument(
        "--fe",
        default=['all'],
        nargs='+',
        type=str,
        help='feature engineering data file path (ex) 00'
    )
    
    parser.add_argument(
        '--valid_num',
        default=1,
        type=int,
        help = '밸리드용 문제를 몇개를 둘 건지'
    )

    parser.add_argument(
        "--file_name", default="train_data.csv", type=str, help="train file name"
    )

    parser.add_argument(
        "--model_dir", default="models/", type=str, help="model directory"
    )
    parser.add_argument(
        "--model_name", default="model.pt", type=str, help="model file name"
    )

    parser.add_argument(
        "--output_dir", default="output/", type=str, help="output directory"
    )
    parser.add_argument(
        "--test_file_name", default="test_data.csv", type=str, help="test file name"
    )

    parser.add_argument(
        '--merge', default=False
    )
    parser.add_argument(
        '--new', default=False
    )
    parser.add_argument("--max_seq_len", default=70, type=int, help="max sequence length" )
    parser.add_argument("--num_workers", default=8, type=int, help="number of workers")
    parser.add_argument('--leak', default=0, type=int)
    # parser.add_argument('--emb_separate', default=0, type=int, help='embedding 을 각 피처의 nunique 값으로 세분화함.')
    # 모델
    parser.add_argument('--loss', default='bce', type=str)
    parser.add_argument('--cate_emb_dim', default=16, type=int) # 라벨인코딩하면 어차피 만개 넘어가서 32 ~ 64 등
    parser.add_argument('--cate_proj_dim', default=16, type=int) # 라벨인코딩하면 어차피 만개 넘어가서 32 ~ 64 등
    parser.add_argument("--cont_proj_dim", default=8, type=int) # 수치형 피처 수보다 적어야 좋을듯. 아니야 riid 는 더 크게 해.
    parser.add_argument(
        "--hidden_dim", default=64, type=int, help="hidden dimension size"
    ) # 위의 두개를 합친거 보다 작아야 좋은.

    parser.add_argument("--n_layers", default=2, type=int, help="number of layers")
    parser.add_argument("--n_heads", default=2, type=int, help="number of heads")
    parser.add_argument("--drop_out", default=0.4, type=float, help="drop out rate")

    # 훈련
    parser.add_argument("--n_epochs", default=300, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=256, type=int, help="batch size")
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
    parser.add_argument("--clip_grad", default=0.75, type=float, help="clip grad")
    parser.add_argument("--patience", default=5, type=int, help="for early stopping")

    parser.add_argument(
        "--log_steps", default=10, type=int, help="print log per n steps"
    )

    ### 중요 ###
    # 일단 보류.
    parser.add_argument("--model", default="GRU", type=str, help="model type")
    parser.add_argument("--optimizer", default="adam", type=str, help="optimizer type")
    parser.add_argument("--scheduler", default="plateau", type=str, help="scheduler type")

    args = parser.parse_args()

    return args
