import numpy as np
import matplotlib.pyplot as plt
import json as js
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging

from data_generator import dict2namespace
from data_loader import load_data
from models import get_model, mkdir


def train(loader, model, optimizer):
    model.train()
    losses = []
    for i, dataset in enumerate(loader):

        batch_loss = model.count_loss(dataset)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        losses.append(batch_loss.detach().numpy())

        if i % 500 == 0:
            plt.clf()
            plt.plot(np.array(losses))
            plt.title(i)
            plt.show(block=False)
            plt.pause(0.00001)

        if torch.isnan(batch_loss):
            print("nan")
            exit()
    return np.mean(np.array(losses))


def test(loader, model):
    model.eval()
    losses = []
    for i, dataset in enumerate(loader):
        batch_loss = model.count_loss(dataset)
        losses.append(batch_loss.detach().numpy())
    return np.mean(np.array(losses))


if __name__ == "__main__":
    with open("./settings/trainset.json") as f:
        args = js.load(f)
    args = dict2namespace(args)

    train_loader, test_loader, valid_loader = load_data(
        args.batch_size, args.data_type,
        args.train_type, args.predict_length
    )

    model = get_model(args)
    dirname = mkdir(args)

    logging.basicConfig(
        filename=dirname+"/log.txt", filemode="w",
        format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
        datefmt="%d-%M-%Y %H:%M:%S", level=logging.INFO
    )

    lr = 0.01
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    loss = []
    test_loss = []
    for i in range(args.epoch_num):
        loss.append(train(train_loader, model, optimizer))

        # 保存单个epoch中的预测误差值
        if i % 5 == 0:
            plt.savefig(dirname+f"/figures/epoch{i}_train.jpg")

        test_loss.append(test(test_loader, model))
        logging.info(f"epoch:{i}, train_loss:{loss[-1]}, " +
                     f"test_loss:{test_loss[-1]}")

        if (i+1) % args.save_epoch_num == 0:
            torch.save(model.state_dict(), dirname+f"/state_dict/epoch{i}.pi")

        plt.clf()
        plt.subplot(1, 2, 1)
        plt.plot(loss)
        plt.subplot(1, 2, 2)
        plt.plot(test_loss)
        plt.savefig(dirname+"/total_loss.jpg")
