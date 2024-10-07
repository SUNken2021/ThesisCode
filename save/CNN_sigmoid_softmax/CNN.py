import torch
from torch import nn
import torch.nn.functional as F
# import numpy as np


class CNN(nn.Module):
    def __init__(self, args) -> None:
        super(CNN, self).__init__()
        self.args = args
        # 输入(1, time, kspace_size)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=args.conv1_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )  # 输出(conv1_channels, time, kspace_size)

        # 输入(conv1_channels/4, time, kspace_size)
        self.conv2 = nn.Conv2d(
            in_channels=args.conv1_channels//4,
            out_channels=args.conv2_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )  # 输出(conv2_channels, time, kspace_size)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.linear1 = nn.Linear(args.conv2_channels//4, 1)
        self.linear2 = nn.Linear(args.input_size, args.output_size)
        self.actication_set(args.acti_name)
        self.end_norm_set(args.end_norm)
        self.loss_count = "mse"
        self.output_size = args.output_size

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
        batch_size, input_size, kspace_size = input.shape
        input = input.reshape(batch_size, 1,
                              input_size, kspace_size)
        mid = self.pool(
            self.acti(
                self.conv1(input).reshape(
                    batch_size, self.args.conv1_channels//4,
                    input_size*2, kspace_size*2
                )
            )
        )
        mid = self.pool(
            self.acti(
                self.conv2(mid).reshape(
                    batch_size, self.args.conv2_channels//4,
                    input_size*2, kspace_size*2
                )
            )
        )
        output = self.end_norm(
            self.linear2(
                self.linear1(
                    mid.reshape(
                        batch_size, self.args.conv2_channels//4, -1
                    ).transpose(1, 2)
                ).reshape(
                    batch_size, input_size, kspace_size
                ).transpose(1, 2)
            ).transpose(1, 2)
        )
        return output

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
    
    def test_loss_short(self, dataset):
        given, predict = dataset
        output = self.forward(given)

        loss = self.loss_fc(output, predict)
        return loss

    def test_loss_long(self, dataset):
        self.input_size = 16
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
