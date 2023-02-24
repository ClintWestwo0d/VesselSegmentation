from __future__ import absolute_import
import torch.nn.functional as F

from .expanding_part_utils import *

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

class DeepPyramid_ResNet50(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, Pyramid_Loss=True):
        super(DeepPyramid_ResNet50, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.Pyramid_Loss = Pyramid_Loss

        self.Backbone = ResNet50_Separate()

        self.firstUpSample = nn.Upsample(scale_factor=2)

        self.glob1 = PVF(input_channels=2048, pool_sizes=[32, 9, 7, 5], dim=32)

        self.glob2 = PVF(input_channels=512, pool_sizes=[64, 9, 7, 5], dim=64)
        self.glob3 = PVF(input_channels=256, pool_sizes=[128, 9, 7, 5], dim=128)
        self.glob4 = PVF(input_channels=64, pool_sizes=[256, 9, 7, 5], dim=256)

        self.up1 = Up(2048 + 1024, 512, [3, 6, 7], bilinear)
        self.up2 = Up(1024, 256, [3, 6, 7], bilinear)
        self.up3 = Up(512, 64, [3, 6, 7], bilinear)
        self.up4 = Up(128, 32, [3, 6, 7], bilinear)

        self.outc = OutConv(32, n_classes)

        if self.Pyramid_Loss:
            print("Pyramid Loss activated in the network")
            self.mask1 = nn.Conv2d(512, 1, kernel_size=3, padding=1)
            self.mask2 = nn.Conv2d(256, 1, kernel_size=3, padding=1)
            self.mask3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        else:
            print("Pyramid Loss is deactivated in the network")
            print("You can activate it by setting Pyramid_Loss=True when initializing the network")

    def forward(self, x):

        x = self.firstUpSample(x)

        out5, out4, out3, out2, out1 = self.Backbone(x)

        out1 = self.glob1(out1)

        x1 = self.up1(out1, out2)
        x1 = self.glob2(x1)

        x2 = self.up2(x1, out3)
        x2 = self.glob3(x2)

        x3 = self.up3(x2, out4)
        x3 = self.glob4(x3)

        x4 = self.up4(x3, out5)

        logits = self.outc(x4)

        if self.Pyramid_Loss:
            mask1 = self.mask1(x1)
            mask2 = self.mask2(x2)
            mask3 = self.mask3(x3)

            return logits, mask1, mask2, mask3

        else:
            return logits

class DeepPyramid_VGG16(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, Pyramid_Loss=True):
        super(DeepPyramid_VGG16, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.Pyramid_Loss = Pyramid_Loss

        self.Backbone = VGG16_Separate()

        self.firstUpSample = nn.Upsample(scale_factor=2)

        self.glob1 = PVF(input_channels=2048, pool_sizes=[32, 9, 7, 5], dim=32)

        self.glob2 = PVF(input_channels=512, pool_sizes=[64, 9, 7, 5], dim=64)
        self.glob3 = PVF(input_channels=256, pool_sizes=[128, 9, 7, 5], dim=128)
        self.glob4 = PVF(input_channels=64, pool_sizes=[256, 9, 7, 5], dim=256)

        self.up1 = Up(2048 + 1024, 512, [3, 6, 7], bilinear)
        self.up2 = Up(1024, 256, [3, 6, 7], bilinear)
        self.up3 = Up(512, 64, [3, 6, 7], bilinear)
        self.up4 = Up(128, 32, [3, 6, 7], bilinear)

        self.outc = OutConv(32, n_classes)

        if self.Pyramid_Loss:
            print("Pyramid Loss activated in the network")
            self.mask1 = nn.Conv2d(512, 1, kernel_size=3, padding=1)
            self.mask2 = nn.Conv2d(256, 1, kernel_size=3, padding=1)
            self.mask3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        else:
            print("Pyramid Loss is deactivated in the network")
            print("You can activate it by setting Pyramid_Loss=True when initializing the network")

    def forward(self, x):

        x = self.firstUpSample(x)

        out5, out4, out3, out2, out1 = self.Backbone(x)

        out1 = self.glob1(out1)

        x1 = self.up1(out1, out2)
        x1 = self.glob2(x1)

        x2 = self.up2(x1, out3)
        x2 = self.glob3(x2)

        x3 = self.up3(x2, out4)
        x3 = self.glob4(x3)

        x4 = self.up4(x3, out5)

        logits = self.outc(x4)

        if self.Pyramid_Loss:
            mask1 = self.mask1(x1)
            mask2 = self.mask2(x2)
            mask3 = self.mask3(x3)

            return logits, mask1, mask2, mask3

        else:
            return logits

class UNet_extract_part_PVF_DPR(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, downsize = 2, bilinear=True):
        super(UNet_extract_part_PVF_DPR, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64//downsize)    # plain conv
        self.Conv2 = conv_block(ch_in=64//downsize, ch_out=128//downsize)
        self.Conv3 = conv_block(ch_in=128//downsize, ch_out=256//downsize)   #plain conv
        self.Conv4 = conv_block(ch_in=256//downsize, ch_out=512//downsize)

        self.Up4 = up_conv(ch_in=512//downsize, ch_out=256//downsize)
        self.Up3 = up_conv(ch_in=256//downsize, ch_out=128//downsize)
        self.Up2 = up_conv(ch_in=128//downsize, ch_out=64//downsize)

        self.Conv_1x1 = nn.Conv2d(64//downsize, output_ch, kernel_size=1, stride=1, padding=0)
        # extracting path
        self.glob1 = PVF(input_channels=256, pool_sizes=[8, 7, 5, 3], dim=8)

        self.glob2 = PVF(input_channels=128, pool_sizes=[16, 9, 7, 5], dim=16)
        self.glob3 = PVF(input_channels=64, pool_sizes=[32, 9, 7, 5], dim=32)
        self.glob4 = PVF(input_channels=32, pool_sizes=[64, 9, 7, 5], dim=64)

        self.up1 = Up(256 + 128, 128, [3, 6, 7], bilinear)
        self.up2 = Up(128 + 64, 64, [3, 6, 7], bilinear)
        self.up3 = Up(64 + 32, 32, [3, 6, 7], bilinear)
        self.up4 = Up(128, 32, [3, 6, 7], bilinear)

        self.outc = OutConv(32, 2)


    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)  # 32,64,64

        x2 = self.Maxpool(x1)   # 32,32,32
        x2 = self.Conv2(x2)     # 64,32,32

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        # extracting path
        x4 = self.glob1(x4)

        out1 = self.up1(x4, x3)
        out1 = self.glob2(out1)

        out2 = self.up2(out1, x2)
        out2 = self.glob3(out2)

        out3 = self.up3(out2, x1)
        out3 = self.glob4(out3)

        out = self.outc(out3)
        out = F.softmax(out, dim=1)
        return out

if __name__ == '__main__':
    net = UNet_extract_part_PVF_DPR(1,2).cuda()
    in1 = torch.randn((4,1,64,64)).cuda()
    out = net(in1)
    print(out.size(),out.min(), out.max())
