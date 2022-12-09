from dataset import load_data, prediction
from model import LightGCN_LSTM
from args import parse_args

import os
import torch

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, valid, test, nodes, feature = load_data(args, device)

    model = LightGCN_LSTM(nodes, feature, args, train, valid)
    model.run()

    prediction(model, test)

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)