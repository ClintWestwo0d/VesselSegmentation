import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DCOA_Conv import Dynamic_conv2d


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_dcoa(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_dcoa, self).__init__()
        self.conv = nn.Sequential(
            Dynamic_conv2d(ch_in, ch_out, kernel_size=3, ratio=0.25, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            Dynamic_conv2d(ch_out, ch_out, kernel_size=3, ratio=0.25, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class up_conv_dcoa(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv_dcoa, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Dynamic_conv2d(ch_in, ch_out, kernel_size=3, ratio=0.25, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, down_size = 4):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64//down_size)
        self.Conv2 = conv_block(ch_in=64//down_size, ch_out=128//down_size)
        self.Conv3 = conv_block(ch_in=128//down_size, ch_out=256//down_size)
        self.Conv4 = conv_block(ch_in=256//down_size, ch_out=512//down_size)
        self.Conv5 = conv_block(ch_in=512//down_size, ch_out=1024//down_size)

        self.Up5 = up_conv(ch_in=1024//down_size, ch_out=512//down_size)
        self.Up_conv5 = conv_block(ch_in=1024//down_size, ch_out=512//down_size)

        self.Up4 = up_conv(ch_in=512//down_size, ch_out=256//down_size)
        self.Up_conv4 = conv_block(ch_in=512//down_size, ch_out=256//down_size)

        self.Up3 = up_conv(ch_in=256//down_size, ch_out=128//down_size)
        self.Up_conv3 = conv_block(ch_in=256//down_size, ch_out=128//down_size)

        self.Up2 = up_conv(ch_in=128//down_size, ch_out=64//down_size)
        self.Up_conv2 = conv_block(ch_in=128//down_size, ch_out=64//down_size)

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

class U_Net_DCOA(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, down_size = 4):
        super(U_Net_DCOA, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block_dcoa(ch_in=img_ch, ch_out=64//down_size)
        self.Conv2 = conv_block(ch_in=64//down_size, ch_out=128//down_size)
        self.Conv3 = conv_block(ch_in=128//down_size, ch_out=256//down_size)
        self.Conv4 = conv_block(ch_in=256//down_size, ch_out=512//down_size)
        self.Conv5 = conv_block(ch_in=512//down_size, ch_out=1024//down_size)

        self.Up5 = up_conv(ch_in=1024//down_size, ch_out=512//down_size)
        self.Up_conv5 = conv_block(ch_in=1024//down_size, ch_out=512//down_size)

        self.Up4 = up_conv(ch_in=512//down_size, ch_out=256//down_size)
        self.Up_conv4 = conv_block(ch_in=512//down_size, ch_out=256//down_size)

        self.Up3 = up_conv(ch_in=256//down_size, ch_out=128//down_size)
        self.Up_conv3 = conv_block(ch_in=256//down_size, ch_out=128//down_size)

        self.Up2 = up_conv(ch_in=128//down_size, ch_out=64//down_size)
        self.Up_conv2 = conv_block(ch_in=128//down_size, ch_out=64//down_size)

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
    net = U_Net_DCOA(1, 2).cuda()
    out = net(x)
    print(out.size(), out.min(), out.max())