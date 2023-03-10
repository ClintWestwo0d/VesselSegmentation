# Dynamic-convolution

import torch
import torch.nn as nn
import torch.nn.functional as F


class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO ?????????
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#???batch??????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)# ????????????????????????????????????
        weight = self.weight.view(self.K, -1)

        # ????????????????????????????????? ????????????batch_size???????????????????????????????????????
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output

class conv_block_DConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_DConv, self).__init__()
        self.conv = nn.Sequential(
            # Dynamic_conv2d(ch_in, ch_out, kernel_size=3, ratio=0.25, padding=1),
            Dynamic_conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            # Dynamic_conv2d(ch_out, ch_out, kernel_size=3, ratio=0.25, padding=1),
            Dynamic_conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv_DConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv_DConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # Dynamic_conv2d(ch_in, ch_out, kernel_size=3, ratio=0.25, padding=1),
            Dynamic_conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class U_Net_DConv(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, down_size = 4):
        super(U_Net_DConv, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block_DConv(ch_in=img_ch, ch_out=64//down_size)
        self.Conv2 = conv_block_DConv(ch_in=64//down_size, ch_out=128//down_size)
        self.Conv3 = conv_block_DConv(ch_in=128//down_size, ch_out=256//down_size)
        self.Conv4 = conv_block_DConv(ch_in=256//down_size, ch_out=512//down_size)
        self.Conv5 = conv_block_DConv(ch_in=512//down_size, ch_out=1024//down_size)

        self.Up5 = up_conv_DConv(ch_in=1024//down_size, ch_out=512//down_size)
        self.Up_conv5 = conv_block_DConv(ch_in=1024//down_size, ch_out=512//down_size)

        self.Up4 = up_conv_DConv(ch_in=512//down_size, ch_out=256//down_size)
        self.Up_conv4 = conv_block_DConv(ch_in=512//down_size, ch_out=256//down_size)

        self.Up3 = up_conv_DConv(ch_in=256//down_size, ch_out=128//down_size)
        self.Up_conv3 = conv_block_DConv(ch_in=256//down_size, ch_out=128//down_size)

        self.Up2 = up_conv_DConv(ch_in=128//down_size, ch_out=64//down_size)
        self.Up_conv2 = conv_block_DConv(ch_in=128//down_size, ch_out=64//down_size)

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
    net = U_Net_DConv(1, 2).cuda()
    out = net(x)
    print(out.size(), out.min(), out.max())