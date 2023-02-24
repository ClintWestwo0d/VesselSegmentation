import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function


class asign_index(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kernel, guide_feature):
        ctx.save_for_backward(kernel, guide_feature)
        guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True),
                                                              1).unsqueeze(2)  # B x 3 x 1 x 25 x 25
        return torch.sum(kernel * guide_mask, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        kernel, guide_feature = ctx.saved_tensors
        guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True),
                                                              1).unsqueeze(2)  # B x 3 x 1 x 25 x 25
        grad_kernel = grad_output.clone().unsqueeze(1) * guide_mask  # B x 3 x 256 x 25 x 25
        grad_guide = grad_output.clone().unsqueeze(1) * kernel  # B x 3 x 256 x 25 x 25
        grad_guide = grad_guide.sum(dim=2)  # B x 3 x 25 x 25
        softmax = F.softmax(guide_feature, 1)  # B x 3 x 25 x 25
        grad_guide = softmax * (grad_guide - (softmax * grad_guide).sum(dim=1, keepdim=True))  # B x 3 x 25 x 25
        return grad_kernel, grad_guide


def xcorr_slow(x, kernel, kwargs):
    """for loop to calculate cross correlation
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, px.size()[0], px.size()[1], px.size()[2])
        pk = pk.view(-1, px.size()[1], pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk, **kwargs)
        out.append(po)
    out = torch.cat(out, 0)
    return out


def xcorr_fast(x, kernel, kwargs):
    """group conv2d to calculate cross correlation
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk, **kwargs, groups=batch, padding=1)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po


class Corr(Function):
    @staticmethod
    def symbolic(g, x, kernel, groups):
        return g.op("Corr", x, kernel, groups_i=groups)

    @staticmethod
    def forward(self, x, kernel, groups, kwargs):
        """group conv2d to calculate cross correlation
        """
        batch = x.size(0)
        channel = x.size(1)
        x = x.view(1, -1, x.size(2), x.size(3))
        kernel = kernel.view(-1, channel // groups, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, **kwargs, groups=groups * batch, padding=1)
        out = out.view(batch, -1, out.size(2), out.size(3))
        return out


class Correlation(nn.Module):
    use_slow = True

    def __init__(self, use_slow=None):
        super(Correlation, self).__init__()
        if use_slow is not None:
            self.use_slow = use_slow
        else:
            self.use_slow = Correlation.use_slow

    def extra_repr(self):
        if self.use_slow: return "xcorr_slow"
        return "xcorr_fast"

    def forward(self, x, kernel, **kwargs):
        if self.training:
            if self.use_slow:
                return xcorr_slow(x, kernel, kwargs)
            else:
                return xcorr_fast(x, kernel, kwargs)
        else:
            return Corr.apply(x, kernel, 1, kwargs)

# Dynamic Region-Aware Convolution
class DRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, region_num=8, **kwargs):
        super(DRConv2d, self).__init__()
        self.region_num = region_num

        self.conv_kernel = nn.Sequential(
            nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
            nn.Conv2d(in_channels, region_num * region_num, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(region_num * region_num, region_num * in_channels * out_channels, kernel_size=1,
                      groups=region_num)
        )
        self.conv_guide = nn.Conv2d(in_channels, region_num, kernel_size=kernel_size, **kwargs, padding=1)

        self.corr = Correlation(use_slow=False)
        self.kwargs = kwargs
        self.asign_index = asign_index.apply

    def forward(self, input):
        kernel = self.conv_kernel(input)
        kernel = kernel.view(kernel.size(0), -1, kernel.size(2), kernel.size(3))  # B x (r*in*out) x W X H
        output = self.corr(input, kernel, **self.kwargs)  # B x (r*out) x W x H
        output = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3))  # B x r x out x W x H
        guide_feature = self.conv_guide(input)
        output = self.asign_index(output, guide_feature)
        return output



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

class conv_block_DR(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_DR, self).__init__()
        self.conv = nn.Sequential(
            DRConv2d(ch_in, ch_out, kernel_size=3, region_num=8),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            DRConv2d(ch_out, ch_out, kernel_size=3, region_num=8),
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

class up_conv_DR(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv_DR, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            DRConv2d(ch_in, ch_out, kernel_size=3, region_num=8),
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

class U_Net_DR(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, down_size = 4):
        super(U_Net_DR, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block_DR(ch_in=img_ch, ch_out=64//down_size)
        self.Conv2 = conv_block_DR(ch_in=64//down_size, ch_out=128//down_size)
        self.Conv3 = conv_block_DR(ch_in=128//down_size, ch_out=256//down_size)
        self.Conv4 = conv_block_DR(ch_in=256//down_size, ch_out=512//down_size)
        self.Conv5 = conv_block_DR(ch_in=512//down_size, ch_out=1024//down_size)

        self.Up5 = up_conv_DR(ch_in=1024//down_size, ch_out=512//down_size)
        self.Up_conv5 = conv_block_DR(ch_in=1024//down_size, ch_out=512//down_size)

        self.Up4 = up_conv_DR(ch_in=512//down_size, ch_out=256//down_size)
        self.Up_conv4 = conv_block_DR(ch_in=512//down_size, ch_out=256//down_size)

        self.Up3 = up_conv_DR(ch_in=256//down_size, ch_out=128//down_size)
        self.Up_conv3 = conv_block_DR(ch_in=256//down_size, ch_out=128//down_size)

        self.Up2 = up_conv_DR(ch_in=128//down_size, ch_out=64//down_size)
        self.Up_conv2 = conv_block_DR(ch_in=128//down_size, ch_out=64//down_size)

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
    x = torch.rand(8, 1, 64, 64)
    net = U_Net_DR(1, 2)
    out = net(x)
    print(out.size(), out.min(), out.max())

