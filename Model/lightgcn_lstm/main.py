from dataset import load_data, prediction
from model import LightGCN_LSTM

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train, valid, test, nodes = load_data('train_data.csv', 'test_data.csv', device)

embedding = 256
hidden_dim = 512
layers = 2

model = LightGCN_LSTM(nodes, embedding, hidden_dim, layers, train, valid, 3)
model.run()

prediction(model, test)