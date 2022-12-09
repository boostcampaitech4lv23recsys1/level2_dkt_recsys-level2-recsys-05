# ====================================================
# CFG
# ====================================================
class CFG:
    use_cuda_if_available = True
    user_wandb = False
    wandb_kwargs = dict(project="dkt-gcn")

    # data
    basepath = "/opt/ml/input/data/"
    loader_verbose = True
    cate_cols = [
        'assessmentItemID', 'testId', 'KnowledgeTag',
        'assessmentItemID_last', 'testId_first', 'testId_last'
    ]
    cont_cols = [
        'elapsed',
        'accuracy_by_assessment', 'accuracy_by_test', 'accuracy_by_tag',
        'accuracy_by_assessment_last', 'accuracy_by_test_first',
        'accuracy_by_test_last', 
        'prior_ac_count', 'prior_quest_count', 'prior_ac_accuracy',
        'prior_relative_ac_sum', 'prior_relative_accuracy',
        'prior_assessment_frequency', 'prior_test_frequency',
        'prior_tags_frequency', 'diff_time_btw_tags', 'prev_tag_answer',
    ]
    ratio = 0.3

    # dump
    output_dir = "./output/"
    pred_file = "submission.csv"

    # build
    embedding_dim = 32  # int
    num_layers = 2  # int
    alpha = None  # Optional[Union[float, Tensor]]
    build_kwargs = {}  # other arguments
    weight = "./weight/best_model.pt"

    # train
    n_epoch = 1500
    learning_rate = 1e-3
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
