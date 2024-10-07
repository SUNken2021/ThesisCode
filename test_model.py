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


# 计算短时间预测和长时间预测的损失
def short_long_loss(dir_name, state_dict_name):
    with open(dir_name + "/trainset.json") as f:
        args = js.load(f)
    args = dict2namespace(args)

    train_loader, test_loader, valid_loader = load_data(
        args.batch_size, args.data_type,
        "short", args.predict_length
    )

    model = get_model(args)
    model.load_state_dict(torch.load(state_dict_name))

    model.eval()
    losses = []
    for i, dataset in enumerate(valid_loader):
        batch_loss = model.test_loss_short(dataset)
        losses.append(batch_loss.detach().numpy())
    short_loss = np.mean(np.array(losses))
    with open(dir_name + "/short_loss.txt", "w") as f:
        f.write(f"short_loss:{short_loss}")
    print(f"short_loss:{short_loss}")

    train_loader, test_loader, valid_loader = load_data(
        args.batch_size, args.data_type,
        "long", args.predict_length
    )
    model.eval()
    losses = []
    for i, dataset in enumerate(valid_loader):
        batch_loss = model.test_loss_long(dataset)
        losses.append(batch_loss.detach().numpy())
    long_loss = np.mean(np.array(losses), axis=0)
    print(long_loss)
    
    plt.plot(long_loss)
    plt.xlabel("Step")
    plt.ylabel("Error")
    plt.savefig(dir_name + "/long_loss.jpg")
    plt.show()


if __name__ == "__main__":
    dir_name = "./save/Transformer_6_12_8_23_37"
    state_dict_name = dir_name + "/state_dict/epoch247.pi"
    short_long_loss(dir_name, state_dict_name)
