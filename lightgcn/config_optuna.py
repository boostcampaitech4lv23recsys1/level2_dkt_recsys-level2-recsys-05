# ====================================================
# CFG
# ====================================================
class CFG:
    use_cuda_if_available = True
    user_wandb = True
    wandb_kwargs = dict(project="dkt-gcn")

    # data
    basepath = "/opt/ml/input/data/"
    loader_verbose = True

    # dump
    output_dir = "./output/"
    pred_file = "submission.csv"

    # build
    embedding_dim = 123 #64  # int
    num_layers = 9 #1  # int
    alpha = 0.0797271712810023  # Optional[Union[float, Tensor]]
    build_kwargs = {}  # other arguments
    weight = "./weight/model_0.8343592104964914_996.pt"
    # train
    n_epoch = 2711 #20
    learning_rate = 0.00217008368273351 #0.001
    weight_basepath = "./weight"
    # {'embedding_dim': 80, 'num_layers': 4, 'alpha': 0.1495598597240675, 'n_epoch': 2334, 'learning_rate': 0.07941647815932328}
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
            "filename": "run_optuna.log",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file_handler"]},
}