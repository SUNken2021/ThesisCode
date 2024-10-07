import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# import numpy as np


class DNN(nn.Module):
    def __init__(self, args):
        super(DNN, self).__init__()
        self.input_size = args.input_size
        self.output_size = args.output_size
        if self.output_size != 1:
            print("output_size!=1")
            exit()

        self.actication_set(args.acti_name)
        self.end_norm_set(args.end_norm)
        self.net_set(args.hidden_layer_num, args.hidden_size)
        self.loss_count = "mse"

    # 各连接层之间的激活函数
    def actication_set(self, acti_name):
        self.acti_name = acti_name
        if acti_name == "relu":
            self.acti = nn.ReLU()
        elif acti_name == "sigmoid":
            self.acti = nn.Sigmoid()
        else:
            print("acti_name error")
            exit()
        return

    def end_norm_set(self, end_norm):
        # 最普通的线性归一化,作为其他归一化的基础
        def norm(input):
            a, b, c = input.shape
            _sum = input.sum(axis=-1)
            _sum = _sum.reshape(a, b, -1)
            _sum = _sum.repeat_interleave(c, dim=-1)
            # 防止和接近0
            _sum = torch.where(_sum == 0, torch.ones_like(_sum), _sum)
            return torch.div(input, _sum)

        if end_norm == "softmax":  # 指数后线性归一化
            self.end_norm = nn.Softmax(dim=-1)
        elif end_norm == "linear":  # 线性归一化
            self.end_norm = norm
        elif end_norm == "relu":  # 取relu后线性归一化
            def relu_norm(input):
                _input = F.relu(input)
                return norm(_input)
            self.end_norm = relu_norm
        elif end_norm == "leaky relu":
            def leaky_relu_norm(input):
                _input = F.leaky_relu(input, 0.01)
                return norm(_input)
            self.end_norm = leaky_relu_norm
        elif end_norm == "mix":  # 指数和relu的结合归一化
            def mix_norm(input):
                _input = torch.exp(input) * (input >= 0) +\
                        (input + 1) * ((input < 0) & (input >= -1)) +\
                        0 * (input < -1)
                return norm(_input)
            self.end_norm = mix_norm
        elif end_norm == "none":  # 不做归一化
            def none_norm(input):
                return input
            self.end_norm = none_norm
        else:
            print("end_norm error")
            exit()
        return

    # 设置DNN神经网络
    def net_set(self, hidden_layer_num, hidden_size):
        self.net = nn.Sequential()
        for i in range(hidden_layer_num):
            # 线性层
            _in = hidden_size if i != 0 else self.input_size
            _out = hidden_size if i != hidden_layer_num-1 else self.output_size
            self.net.add_module(f"hidden{i}", nn.Linear(_in, _out))
            # 标准化层和激活层
            if i != hidden_layer_num-1:
                self.net.add_module("layernorm", nn.LayerNorm(hidden_size))
                self.net.add_module(self.acti_name, self.acti)
        return

    def forward(self, input):
        batch_size, _input_size, kspace_size = input.shape
        # 判断输入的时间长度是否与模型的input大小一致
        if _input_size != self.input_size:
            print("input_size should be {}, but is {}".format(
                self.input_size, _input_size))

        # 各动量态分别进行预测
        prediction = torch.zeros([batch_size, self.output_size, kspace_size])
        for i in range(kspace_size):
            prediction[:, :, i] = self.net(input[:, :, i]) + input[:, -1:, i]

        prediction = self.end_norm(prediction)  # 归一化
        return prediction

    def count_loss(self, dataset):
        given, predict = dataset
        output = self.forward(given)

        loss = self.loss_fc(output, predict)
        return loss

    def loss_fc(self, output, predict):
        if self.loss_count == "mse":
            _fc = nn.MSELoss()
            loss = _fc(output, predict)/self.output_size
        elif self.loss_count == "kl":
            kspace_size = predict.shape[-1]
            loss = F.kl_div(torch.log(output.reshape(-1, kspace_size)),
                            predict.reshape(-1, kspace_size),
                            reduction="batchmean")
        return loss

    # 返回单步预测的误差
    def test_loss_short(self, dataset):
        given, predict = dataset
        output = self.forward(given)

        loss = self.loss_fc(output, predict)
        return loss

    # 返回多步预测的误差
    def test_loss_long(self, dataset):
        data = dataset[0]
        given, predict = data[:, :self.input_size, :], \
            data[:, self.input_size:, :]
        batch_size, _input_size, kspace_size = given.shape
        N = predict.shape[1]
        prediction = torch.zeros([batch_size, N, kspace_size])
        for i in range(N):
            if i <= self.input_size:
                input = torch.cat(
                    (given[:, i:, :], prediction[:, :i, :]), dim=1
                )
            else:
                input = prediction[:, i-self.input_size:i, :]
            prediction[:, i:i+1, :] = self.forward(input)

        loss = torch.square(prediction - predict)
        return loss.mean([0, -1])
