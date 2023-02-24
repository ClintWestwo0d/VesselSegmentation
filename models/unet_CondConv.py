import functools
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


class _routing(nn.Module):

    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        x = torch.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return torch.sigmoid(x)


class CondConv2D(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', num_experts=3, dropout_rate=0.2):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(CondConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        self._routing_fn = _routing(in_channels, num_experts, dropout_rate)

        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))

        self.reset_parameters()

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, inputs):
        b, _, _, _ = inputs.size()
        res = []
        for input in inputs:
            input = input.unsqueeze(0)
            pooled_inputs = self._avg_pooling(input)
            routing_weights = self._routing_fn(pooled_inputs)
            kernels = torch.sum(routing_weights[:, None, None, None, None] * self.weight, 0)
            out = self._conv_forward(input, kernels)
            res.append(out)
        return torch.cat(res, dim=0)


class conv_block_Condconv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_Condconv, self).__init__()
        self.conv = nn.Sequential(
            # Dynamic_conv2d(ch_in, ch_out, kernel_size=3, ratio=0.25, padding=1),
            CondConv2D(ch_in, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            # Dynamic_conv2d(ch_out, ch_out, kernel_size=3, ratio=0.25, padding=1),
            CondConv2D(ch_out, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv_Condconv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv_Condconv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # Dynamic_conv2d(ch_in, ch_out, kernel_size=3, ratio=0.25, padding=1),
            CondConv2D(ch_in, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Model(nn.Module):
    def __init__(self, num_experts):
        super(Model, self).__init__()
        self.condconv2d = CondConv2D(1, 128, kernel_size=3)

    def forward(self, x):
        x = self.condconv2d(x)


class U_Net_CondConv(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, down_size = 4):
        super(U_Net_CondConv, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block_Condconv(ch_in=img_ch, ch_out=64//down_size)
        self.Conv2 = conv_block_Condconv(ch_in=64//down_size, ch_out=128//down_size)
        self.Conv3 = conv_block_Condconv(ch_in=128//down_size, ch_out=256//down_size)
        self.Conv4 = conv_block_Condconv(ch_in=256//down_size, ch_out=512//down_size)
        self.Conv5 = conv_block_Condconv(ch_in=512//down_size, ch_out=1024//down_size)

        self.Up5 = up_conv_Condconv(ch_in=1024//down_size, ch_out=512//down_size)
        self.Up_conv5 = conv_block_Condconv(ch_in=1024//down_size, ch_out=512//down_size)

        self.Up4 = up_conv_Condconv(ch_in=512//down_size, ch_out=256//down_size)
        self.Up_conv4 = conv_block_Condconv(ch_in=512//down_size, ch_out=256//down_size)

        self.Up3 = up_conv_Condconv(ch_in=256//down_size, ch_out=128//down_size)
        self.Up_conv3 = conv_block_Condconv(ch_in=256//down_size, ch_out=128//down_size)

        self.Up2 = up_conv_Condconv(ch_in=128//down_size, ch_out=64//down_size)
        self.Up_conv2 = conv_block_Condconv(ch_in=128//down_size, ch_out=64//down_size)

        self.Conv_1x1 = nn.Conv2d(64//down_size, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = F.softmax(d1,dim=1)  # mine

        return d1

if __name__ == '__main__':
    x = torch.rand(4, 1, 64, 64).cuda()
    net = U_Net_CondConv(1, 2).cuda()
    out = net(x)
    print(out.size(), out.min(), out.max())