# ====================================================
# CFG
# ====================================================
class CFG:
    use_cuda_if_available = True
    user_wandb = True
    wandb_kwargs = dict(project="dkt-gcn")

    # data
    # basepath = "/opt/ml/input/data/"
    basepath = "/opt/ml/input/JMKIM/"
    loader_verbose = True

    # dump
    output_dir = "./output/"
    pred_file = "submission.csv"

    # build
    embedding_dim = 32 #64  # int
    num_layers = 2 #1  # int
    alpha = None  # Optional[Union[float, Tensor]]
    build_kwargs = {}  # other arguments
    weight = "./weight/best_model.pt"
    
    # train
    n_epoch = 1500 #20
    learning_rate = 1e-3 #0.001
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
            "filename": "run_optuna.log",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file_handler"]},
}