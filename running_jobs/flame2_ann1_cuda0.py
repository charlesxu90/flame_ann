from torch.nn.modules.loss import _Loss
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.utils.data as Data
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import argparse
import logging


class FlameLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'sum') -> None:
        super(FlameLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.sum((target + 1) * (input - target) ** 2)


def create_model(n_input, n_output, nodes, device='cpu'):
    modules = []

    # add first hidden layer (and input layer)
    modules.append(torch.nn.Linear(n_input, nodes[0]))
    modules.append(torch.nn.LeakyReLU())

    for i in range(1, len(nodes)):
        modules.append(torch.nn.Linear(nodes[i-1], nodes[i]))
        modules.append(torch.nn.LeakyReLU())

    # add output layer
    modules.append(torch.nn.Linear(nodes[-1], n_output))

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

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    best_loss = float('inf')
    writer = SummaryWriter(save_path)

    for epoch in range(n_epoch):
        for step, (batch_x, batch_y) in enumerate(loader):
            b_x = Variable(batch_x).to(device)
            b_y = Variable(batch_y).to(device)

            y_pred = model(b_x.float())
            loss = FlameLoss()(y_pred, b_y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % eval_interval == 0:
            # evaluate model every $eval_interval$ steps
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test.float())
                e_loss = torch.nn.MSELoss(reduction='sum')(y_pred, y_test.float())

                y_err = (y_pred-y_test).cpu().data.numpy()
                y_abs_err = np.abs(y_err).max(axis=0)   # CO_err, CO2_err
                logging.info(f"epoch {epoch+1}: Train loss={loss:.6f}, Eval loss={e_loss:.6f}, "
                             + f"CO_err={y_abs_err[0]:.6f}, CO2_err={y_abs_err[1]:.6f}")
                writer.add_scalar('Eval loss', e_loss, epoch + 1)
                writer.add_scalar('CO err', y_abs_err[0], epoch + 1)
                writer.add_scalar('CO2 err', y_abs_err[1], epoch + 1)

                if e_loss < best_loss:
                    best_loss = e_loss
                    save_model(model, save_path + f"model_{epoch+1}_{e_loss:.6f}.pt")

            model.train()

    save_model(model, save_path + f"model_final_{e_loss:.6f}.pt")


def dl_modeling(X_train, y_train, X_test, y_test, nodes, ckpt_path=None, device='cpu', n_epoch=100):
    model = create_model(X_train.shape[1], y_train.shape[1], nodes, device=device)

    if ckpt_path is not None:
        model = load_model(model, ckpt_path)
    
    train(model, X_train, y_train, X_test, y_test, n_epoch=n_epoch, device=device, save_path='./ann1_models/', eval_interval=1)


def main(args):
    data_dir = args.data_dir
    device = args.device

    logging.basicConfig(filename='./ann1_models/train.log', filemode='w',
                        format='%(asctime)s - %(message)s', level=logging.INFO)

    if args.test == 1:
        df_train = pd.read_pickle(data_dir + "train2_norm_sampled_test_0.05.pkl")
        epochs = 1
    else:
        df_train = pd.read_pickle(data_dir + "train2_norm_sampled_train_0.95.pkl")
        epochs = args.epochs

    df_test = pd.read_pickle(data_dir + "train2_norm_sampled_test_0.05.pkl")

    X_train = torch.tensor(df_train[["CO+CO2", "H*", "O*", "C*"]].values, device=device)
    y_train = torch.tensor(df_train[["CO", "CO2"]].values, device=device)
    X_test = torch.tensor(df_test[["CO+CO2", "H*", "O*", "C*"]].values, device=device)
    y_test = torch.tensor(df_test[["CO", "CO2"]].values, device=device)

    dl_modeling(X_train, y_train, X_test, y_test, args.nodes, ckpt_path=args.ckpt, device=device, n_epoch=epochs)

def parse_args():
    parser = argparse.ArgumentParser(description="Python script to run DL model")
    parser.add_argument("--data_dir", type=str, help="Path to the folder containing training/testing data")
    parser.add_argument("--nodes", type=int, nargs='+', default=[200,50,200,100],
                        help="Nodes in hidden layers, default [200,50,200,100]")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use by pytorch, default cuda")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs in Pytorch, default 100")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint to initiate training")
    parser.add_argument("--test", type=int, default=0, help="1 for test, default 0")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
