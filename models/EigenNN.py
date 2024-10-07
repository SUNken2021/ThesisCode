import torch
from torch import nn
import torch.nn.functional as F
# import numpy as np


class Encoder(nn.Module):
    def __init__(self, args) -> None:
        super(Encoder, self).__init__()
        self.args = args


class Decoder(nn.Module):
    def __init__(self, args) -> None:
        super(Decoder, self).__init__()
        self.args = args
        self.input_size
        self.actication_set(args.acti_name)
        self.end_norm_set(args.end_norm)
        self.net_set(args.hidden_layer_num, args.hidden_size)

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
        batch_size, tlength, eigen_size = input.shape

        # 投影
        output = torch.zeros([batch_size, tlength, self.space_size])
        for i in range(tlength):
            prediction[:, :, i] = self.net(input[:, :, i]) + input[:, -1:, i]

        prediction = self.end_norm(prediction)  # 归一化
        return prediction


class EigenNN(nn.Module):
    def __init__(self, args) -> None:
        super(EigenNN, self).__init__()
        self.args = args

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

    def forward(self, input):
        pass

    def count_loss(self, dataset):
        pass

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

    def test_loss_short(self, dataset):
        pass

    def test_loss_long(self, dataset):
        pass
