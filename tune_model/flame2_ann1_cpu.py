from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from torch import Tensor
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np
import pandas as pd
import torch
import datetime
import argparse


class FlameLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'sum') -> None:
        super(FlameLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.sum((target + 1) * (input - target) ** 2)


def create_model(n_input, n_output, layers, device='cpu'):
    modules = []

    # add first hidden layer (and input layer)
    modules.append(torch.nn.Linear(n_input, layers[0]))
    modules.append(torch.nn.LeakyReLU())

    for i in range(1, len(layers)):
        modules.append(torch.nn.Linear(layers[i-1], layers[i]))
        modules.append(torch.nn.LeakyReLU())

    # add output layer
    modules.append(torch.nn.Linear(layers[-1], n_output))

    model = torch.nn.Sequential(*modules).to(device)

    return model


def train(model, X_train, y_train, X_test, y_test, device='cpu',
          n_epoch=100, lr=0.001, decayRate=0.96, BATCH_SIZE=256):

    # Process training data
    torch.manual_seed(1)    # reproducible
    train_dataset = Data.TensorDataset(X_train, y_train)
    loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False, )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, decayRate)  # lr*gamma^step

    # print(f"Start time = {datetime.datetime.now()}")
    for epoch in range(n_epoch):
        for step, (batch_x, batch_y) in enumerate(loader):
            b_x = Variable(batch_x).to(device)
            b_y = Variable(batch_y).to(device)

            y_pred = model(b_x.float())
            loss = FlameLoss()(y_pred, b_y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # lr_decay.step()

    # evaluate model:
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test.float())
        mseLoss = torch.nn.MSELoss(reduction='sum')(y_pred, y_test.float())

    y_err = (y_pred-y_test).cpu().data.numpy()
    std_err = np.abs(y_err).max(axis=0)   # CO_err, CO2_err
    return mseLoss, std_err[0], std_err[1]


def dl_modeling(X_train, y_train, X_test, y_test, layers, n_epoch=100):
    model = create_model(X_train.shape[1], y_train.shape[1], layers)
    return train(model, X_train, y_train, X_test, y_test, n_epoch=n_epoch)


def main(args):
    data_dir = args.data_dir
    df_train = pd.read_pickle(data_dir + "train2_norm_sampled_100k.pkl")
    df_test = pd.read_pickle(data_dir + "train2_norm_sampled_100k.pkl")


    X_train = torch.tensor(df_train[["CO+CO2", "H*", "O*", "C*"]].values, device='cpu')
    y_train = torch.tensor(df_train[["CO", "CO2"]].values, device='cpu')
    X_test = torch.tensor(df_test[["CO+CO2", "H*", "O*", "C*"]].values, device='cpu')
    y_test = torch.tensor(df_test[["CO", "CO2"]].values, device='cpu')

    loss, CO_err, CO2_err = dl_modeling(X_train, y_train, X_test, y_test, args.nodes, n_epoch=args.epochs)
    node_str = '\t'.join([str(x) for x in args.nodes])
    print(f"{node_str}\t{loss:.4f}\t{CO_err:.4f}\t{CO2_err:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description="Python script to tune DL model layers")
    parser.add_argument("--data_dir", type=str, help="Path to the folder containing training data")
    parser.add_argument("--nodes", type=int, nargs='+', default=[10,10,10],
                        help="Nodes in hidden layers, default [10,10,10]")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs in Pytorch, default 100")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)