import numpy as np
import matplotlib.pyplot as plt
import json as js
import argparse

import torch
from torch.utils.data import TensorDataset, DataLoader


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


# 将长的时间预测数据划分为给定数据和预测数据
def subdivide(args, raw_data):
    given, predict = args.given, args.predict
    num, tlength, _size = raw_data.shape
    repetition_divide = args.repetition_divide  # 选取初始点的间隔
    _num = (tlength-given-predict)//repetition_divide  # 单条数据能够取的初始点数量

    given_data = torch.zeros([
        num*_num, given, _size
    ])
    predict_data = torch.zeros([
        num*_num, predict, _size
    ])
    for i in range(num):
        for j in range(_num):
            _j = j*repetition_divide
            given_data[i*_num+j, :, :] = raw_data[i, _j:_j+given, :]
            predict_data[i*_num+j, :, :] = raw_data[
                i, _j+given:_j+given+predict, :]
    return given_data, predict_data


# 将长的时间预测数据划分为给定数据和预测数据
def divide(args, train, test, valid):
    train_given, train_predict = subdivide(args, train)
    test_given, test_predict = subdivide(args, test)
    valid_given, valid_predict = subdivide(args, valid)
    return [train_given, train_predict,
            test_given, test_predict,
            valid_given, valid_predict]


if __name__ == "__main__":
    with open("./settings/divider.json") as f:
        args = js.load(f)
    args = dict2namespace(args)

    noninteraction_data = np.load("./data/noninteraction_data.npy")
    interaction_data = np.load("./data/interaction_data.npy")

    non_data = torch.from_numpy(noninteraction_data).to(torch.float32)
    _data = torch.from_numpy(interaction_data).to(torch.float32)

    # 数据打乱
    idx = torch.randperm(non_data.shape[0])
    non_data = non_data[idx].view(non_data.size())
    idx = torch.randperm(_data.shape[0])
    _data = _data[idx].view(_data.size())
    # 划分训练集，测试集和验证集
    n_data = non_data.shape[0]
    _train, _test, _valid = 8, 1, 1
    _total = _train + _test + _valid
    divition0, divition1 = n_data*_train//_total, \
        n_data*(_train+_test)//_total
    # 无相互作用、有相互作用和混合数据集
    non_train_data, non_test_data, non_valid_data = non_data[:divition0], \
        non_data[divition0:divition1], non_data[divition1:]
    _train_data, _test_data, _valid_data = _data[:divition0], \
        _data[divition0:divition1], _data[divition1:]
    mix_train_data, mix_test_data, mix_valid_data = torch.cat(
        [non_train_data, _train_data], dim=0
    ), torch.cat(
        [non_test_data, _test_data], dim=0
    ), torch.cat(
        [non_valid_data, _valid_data], dim=0
    )
    # 保存长预测数据
    torch.save(
        [non_train_data, non_test_data, non_valid_data],
        "./data/noninteraction_data_long.pt"
    )
    torch.save(
        [_train_data, _test_data, _valid_data],
        "./data/interaction_data_long.pt"
    )
    torch.save(
        [mix_train_data, mix_test_data, mix_valid_data],
        "./data/mix_data_long.pt"
    )
    # 保存短预测数据
    torch.save(
        divide(args, non_train_data, non_test_data, non_valid_data),
        f"./data/noninteraction_data_{args.predict}.pt"
    )
    torch.save(
        divide(args, _train_data, _test_data, _valid_data),
        f"./data/interaction_data_{args.predict}.pt"
    )
    torch.save(
        divide(args, mix_train_data, mix_test_data, mix_valid_data),
        f"./data/mix_data_{args.predict}.pt"
    )
