from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from torch import Tensor
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np
import pandas as pd
import torch
import argparse
import logging


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


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


def load_model(model, ckpt_path):
    model.load_state_dict(torch.load(ckpt_path))
    return model


def train(model, X_train, y_train, X_test, y_test, device='cpu',
          n_epoch=100, lr=0.001, momentum=0.9, BATCH_SIZE=128, save_path='./', eval_interval=1):

    # Process training data
    torch.manual_seed(1)    # reproducible
    train_dataset = Data.TensorDataset(X_train, y_train)
    loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False, )

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)  # only works with mean, not sum, loss 
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    best_loss = float('inf')

    for epoch in range(n_epoch):
        for step, (batch_x, batch_y) in enumerate(loader):
            b_x = Variable(batch_x).to(device)
            b_y = Variable(batch_y).to(device)

            y_pred = model(b_x.float())
            loss = torch.nn.MSELoss(reduction='mean')(y_pred, b_y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % eval_interval == 0:
            # evaluate model every $eval_interval$ steps
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test.float())
                e_loss = torch.nn.MSELoss(reduction='mean')(y_pred, y_test.float())

                y_err = (y_pred-y_test).cpu().data.numpy()
                y_abs_err = np.abs(y_err).max(axis=0)
                logging.info(f"epoch {epoch+1}: Train loss={loss:.4f}, Eval loss={e_loss:.4f}, Max_err={max(y_abs_err):.4f}, M1_err={y_abs_err[0]:.4f}, M2_err={y_abs_err[1]:.4f}")
                if e_loss < best_loss:
                    best_loss = e_loss
                    save_model(model, save_path + f"model_{epoch+1}_{e_loss:.4f}.pt")

            model.train()

    save_model(model, save_path + f"model_final_{loss:.4f}.pt")


def dl_modeling(X_train, y_train, X_test, y_test, nodes, ckpt_path=None, device='cpu', n_epoch=100):
    model = create_model(X_train.shape[1], y_train.shape[1], nodes, device=device)

    if ckpt_path is not None:
        model = load_model(model, ckpt_path)

    train(model, X_train, y_train, X_test, y_test, n_epoch=n_epoch, device=device, save_path='./ann2_models/', eval_interval=1)


def main(args):
    data_dir = args.data_dir
    device = args.device

    logging.basicConfig(filename='./ann2_models/train.log', filemode='w',
                        format='%(asctime)s - %(message)s', level=logging.INFO)
    if args.test == 1:
        df_train = pd.read_pickle(data_dir + "train2_norm_sampled_test_0.05.pkl")
        epochs = 1
    else:
        df_train = pd.read_pickle(data_dir + "train2_norm_sampled_train_0.95.pkl")
        epochs = args.epochs

    df_test = pd.read_pickle(data_dir + "train2_norm_sampled_test_0.05.pkl")

    X_train = torch.tensor(df_train[["CO+CO2", "H*", "O*", "C*"]].values, device=device)
    y_train = torch.tensor(df_train[["n2_out"+ f"{v}" for v in range(1, 51)]].values, device=device)
    X_test = torch.tensor(df_test[["CO+CO2", "H*", "O*", "C*"]].values, device=device)
    y_test = torch.tensor(df_test[["n2_out"+ f"{v}" for v in range(1, 51)]].values, device=device)

    dl_modeling(X_train, y_train, X_test, y_test, args.nodes, ckpt_path=args.ckpt, device=device, n_epoch=epochs)


def parse_args():
    parser = argparse.ArgumentParser(description="Python script to run DL model")
    parser.add_argument("--data_dir", type=str, help="Path to the folder containing training/testing data")
    parser.add_argument("--nodes", type=int, nargs='+', default=[100,200,200,200],
                        help="Nodes in hidden layers, default [100,200,200,200]")
    parser.add_argument("--device", type=str, default='cuda:1', help="Device to use by pytorch, default cuda:1")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs in Pytorch, default 100")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint to initiate training")
    parser.add_argument("--test", type=int, default=0, help="1 for test, default 0")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
