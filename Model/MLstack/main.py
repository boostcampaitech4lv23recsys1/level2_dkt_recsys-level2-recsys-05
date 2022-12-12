from dataset import load_data
from trainer import inference
from args import parse_args

import os
import torch

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_valid, y_train, y_valid, test, columns = load_data(args)

    inference(X_train, X_valid, y_train, y_valid, test, columns)

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)