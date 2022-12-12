import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")


    parser.add_argument(
        "--data_dir",
        default="/opt/ml/input/data/",
        type=str,
        help="data directory",
    )
    parser.add_argument(
        "--asset_dir", default="asset/", type=str, help="data directory"
    )

    parser.add_argument(
        "--file_name", default="train_feature_engineering.csv", type=str, help="train file name"
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
        "--output_file_name", default="submission.csv", type=str, help="output file name"
    )
    parser.add_argument(
        "--test_file_name", default="test_feature_engineering.csv", type=str, help="test file name"
    )

    # 데이터
    parser.add_argument(
        "--cate_cols",
        default=[
            'assessmentItemID', 'testId', 'KnowledgeTag',
            'assessmentItemID_last', 'testId_first', 'testId_last'
        ],
        type=list,
        help="categorical feature names"
    )
    parser.add_argument(
        "--cont_cols",
        default=[
            'elapsed',
            'accuracy_by_assessment', 'accuracy_by_test', 'accuracy_by_tag',
            'accuracy_by_assessment_last', 'accuracy_by_test_first',
            'accuracy_by_test_last', 
            'prior_ac_count', 'prior_quest_count', 'prior_ac_accuracy',
            'prior_relative_ac_sum', 'prior_relative_accuracy',
            'prior_assessment_frequency', 'prior_test_frequency',
            'prior_tags_frequency', 'diff_time_btw_tags', 'prev_tag_answer',
        ],
        type=list,
        help="continuous feature names"
    )
    # 
    
    parser.add_argument("--num_workers", default=1, type=int, help="number of workers")

    # 모델
    

    # 훈련
    
    parser.add_argument("--clip_grad", default=10, type=int, help="clip grad")
    
    parser.add_argument(
        "--log_steps", default=50, type=int, help="print log per n steps"
    )

    ### 중요 ###
    parser.add_argument("--model", default="bert", type=str, help="model type")
    

    args = parser.parse_args()

    return args
