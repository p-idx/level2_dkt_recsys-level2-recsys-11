# ====================================================
# CFG
# ====================================================
class CFG:
    use_cuda_if_available = True
    user_wandb = True
    wandb_kwargs = dict(project="dkt-gcn")

    # data
    basepath = "/opt/ml/level2_dkt_recsys-level2-recsys-11/data/"
    loader_verbose = True

    fe_num = '09'

    # dump
    output_dir = "./output/"
    pred_file = "submission.csv"

    # build
    embedding_dim = 128  # int
    num_layers = 3  # int (보통 5까지 늘리지만 4-10도 학습 가능) - 실험 요소
    alpha = None  # Optional[Union[float, Tensor]]
    build_kwargs = {}  # other arguments
    weight = "./weight/best_model.pt"

    # train
    n_epoch = 2000
    learning_rate = 0.001
    weight_basepath = "./weight"


logging_conf = {  # only used when 'user_wandb==False'
    "version": 1,
    "formatters": {
        "basic": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "basic",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "basic",
            "filename": "run.log",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file_handler"]},
}
