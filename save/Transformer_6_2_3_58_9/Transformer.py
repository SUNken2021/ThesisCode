import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class Transformer(nn.Module):
    def __init__(self, args) -> None:
        super(Transformer, self).__init__()
        self.args = args
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.kspace_size, nhead=1, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=args.kspace_size, nhead=1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.linear = nn.Linear(args.kspace_size, args.kspace_size)
        self.end_norm_set(args.end_norm)
        self.loss_count = "mse"
        pass

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

    def forward(self, encoder_input, decoder_input):
        self.memory = self.encoder(encoder_input)
        mask = nn.Transformer.generate_square_subsequent_mask(
            decoder_input.shape[1])
        output = self.decoder(decoder_input, self.memory, mask)
        output = self.end_norm(self.linear(output))
        return output

    def count_loss(self, dataset):
        given, predict = dataset
        output = self.forward(given, given)
        ground_truth = torch.cat(
            (given[:, 1:, :], predict), dim=1
        )
        # 前几个数据存在较大误差,不考虑进误差修正中
        # 生成分布的kl散度
        loss_fc = nn.MSELoss()
        batch_size, tlength = output.shape[0], output.shape[1]
        loss = loss_fc(ground_truth[:, 1:, :], output[:, 1:, :])/(
            batch_size * (tlength - 1)
        )
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
