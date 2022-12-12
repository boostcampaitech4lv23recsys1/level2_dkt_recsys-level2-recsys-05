import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")

    parser.add_argument(
        "--file_name", default="train_data.csv", type=str, help="train file name"
    )

    parser.add_argument(
        "--output_dir", default="output/", type=str, help="output directory"
    )
    parser.add_argument(
        "--test_file_name", default="test_data.csv", type=str, help="test file name"
    )

    parser.add_argument("--patience", default=3, type=int, help="for early stopping")

    args = parser.parse_args()

    return args
