import torch.nn.functional as F
import os

from models.sk_att_fusion import SKConv_my as SKConv
from models.DCOA_Conv import Dynamic_conv2d
from models.ODConv import ODConv2d
from models.expanding_part_utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"




class encoder(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(encoder, self).__init__()
        self.conv = nn.Sequential(
            Dynamic_conv2d(ch_in, ch_out, kernel_size=3, ratio=0.25, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_odconv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_odconv, self).__init__()
        self.conv = nn.Sequential(
            # Dynamic_conv2d(ch_in, ch_out, kernel_size=3, ratio=0.25, padding=1),
            ODConv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            # Dynamic_conv2d(ch_out, ch_out, kernel_size=3, ratio=0.25, padding=1),
            ODConv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv_odconv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv_odconv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # Dynamic_conv2d(ch_in, ch_out, kernel_size=3, ratio=0.25, padding=1),
            ODConv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
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


class conv_block_dcoa_single(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_dcoa_single, self).__init__()
        self.conv = nn.Sequential(
            Dynamic_conv2d(ch_in, ch_out, kernel_size=3, ratio=0.25, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            # Dynamic_conv2d(ch_out, ch_out, kernel_size=3, ratio=0.25, padding=1),
            # nn.BatchNorm2d(ch_out),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
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



# ==========================Core Module================================
class conv_block_sk(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_sk, self).__init__()
        self.conv = nn.Sequential(
            SKConv(ch_in, ch_in, ch_out),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            SKConv(ch_out, ch_out, ch_out),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv_sk(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv_sk, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            SKConv(ch_in, ch_in, ch_out),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x




class OCENet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, downsize = 2, bilinear=True):
        super(OCENet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64//downsize)    # plain conv
        self.Conv1_dc = conv_block_dcoa(ch_in=64//downsize, ch_out=64//downsize)    # Dynamic Complex Orientation Aware Convolution DCOA Conv
        # self.Conv1_odconv = conv_block_odconv(ch_in=64 // downsize, ch_out=64 // downsize)  # Omni-Dimensional Dynamic Convolution ODConv
        self.Conv1_sk = SKConv(64 // downsize)    #  use SK Attention module to fuse the DCOA features (orientation features) and the plain features
        # self.Conv1_sk = nn.Conv2d(64//downsize, 64//downsize, kernel_size=1)   # the performance will drop if we use conv instead of SK_conv
        self.Fusion1 = conv_block(ch_in=(64 + 64) // downsize, ch_out=64 // downsize)


        self.Conv2 = conv_block(ch_in=64//downsize, ch_out=128//downsize)
        self.Conv2_dc = conv_block_dcoa(ch_in=128//downsize, ch_out=128//downsize)
        # self.Conv2_odconv = conv_block_odconv(ch_in=128 // downsize, ch_out=128 // downsize)
        self.Conv2_sk = SKConv(128//downsize)
        # self.Conv2_sk = nn.Conv2d(128 // downsize, 128 // downsize, kernel_size=1)
        self.Fusion2 = conv_block(ch_in=(128 + 128) // downsize, ch_out=128 // downsize)

        self.Conv3 = conv_block(ch_in=128//downsize, ch_out=256//downsize)   #plain conv
        self.Conv3_dc = conv_block_dcoa(ch_in=256//downsize, ch_out=256//downsize)    # DCOA Conv
        # self.Conv3_odconv = conv_block_odconv(ch_in=256 // downsize, ch_out=256 // downsize)  # ODConv
        self.Conv3_sk = SKConv(256 // downsize)
        # self.Conv3_sk = nn.Conv2d(256 // downsize, 256 // downsize, kernel_size=1)
        self.Fusion3 = conv_block(ch_in=(256 + 256) // downsize, ch_out=256 // downsize)

        self.Conv4 = conv_block(ch_in=256//downsize, ch_out=512//downsize)
        self.Conv4_dc = conv_block_dcoa(ch_in=512//downsize, ch_out=512//downsize)
        # self.Conv4_odconv = conv_block_odconv(ch_in=512 // downsize, ch_out=512 // downsize)
        self.Conv4_sk = SKConv(512//downsize)
        self.Fusion4 = conv_block(ch_in=(512 + 512) // downsize, ch_out=512 //downsize)
        # self.Conv5 = conv_block(ch_in=512//downsize, ch_out=1024//downsize)
        #
        # self.Up5 = up_conv(ch_in=1024//downsize, ch_out=512//downsize)
        # self.Att5 = GLFM(F_g=512 // downsize, F_l=512 // downsize, F_int=256 // downsize)    # Global and Local Fusion Module (GLFM)
        # self.Up_conv5 = conv_block(ch_in=1024//downsize, ch_out=512//downsize)

        self.Up4 = up_conv(ch_in=512//downsize, ch_out=256//downsize)
        self.Up_conv4 = conv_block(ch_in=512//downsize, ch_out=256//downsize)

        self.Up3 = up_conv(ch_in=256//downsize, ch_out=128//downsize)
        self.Up3_dc = up_conv_dcoa(ch_in=256//downsize, ch_out=128//downsize)    # DCOA UP_CONV
        # self.Up3_odconv = up_conv_odconv(ch_in=256 // downsize, ch_out=128 // downsize)  # ODConv UP_CONV
        self.Up3_sk = SKConv(256//downsize)
        # self.Up3_sk = nn.Conv2d(256 // downsize, 256 // downsize, kernel_size=1)
        self.Up_conv3 = conv_block_dcoa(ch_in=256//downsize, ch_out=128//downsize)

        self.Up2 = up_conv(ch_in=128//downsize, ch_out=64//downsize)
        self.Up2_dc = up_conv_dcoa(ch_in=128//downsize, ch_out=64//downsize)
        self.Up2_sk = SKConv(128//downsize)
        # self.Up2_sk = nn.Conv2d(128 // downsize, 128 // downsize, kernel_size=1)

        self.Up_conv2 = conv_block_dcoa(ch_in=128//downsize, ch_out=64//downsize)


        self.Conv_1x1 = nn.Conv2d(64//downsize, output_ch, kernel_size=1, stride=1, padding=0)

        self.conv_prob = nn.Conv2d(64 // downsize, 1, kernel_size=1, stride=1, padding=0)

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
        x1_dcoa = self.Conv1_dc(x1) # 32,64,64
        x1 = self.Conv1_sk(x1, x1_dcoa) # 32,64,64
        # x1_odconv = self.Conv1_odconv(x1)  # 32,64,64
        # x1 = self.Conv1_sk(x1, x1_odconv)  # 1,64,64

        x2 = self.Maxpool(x1)   # 32,32,32
        x2 = self.Conv2(x2)     # 64,32,32
        x2_dcoa = self.Conv2_dc(x2) # 64,32,32
        x2 = self.Conv2_sk(x2, x2_dcoa) # 64,32,32
        # x2_odconv = self.Conv2_odconv(x2)  # 64,32,32
        # x2 = self.Conv2_sk(x2, x2_odconv)  # 64,32,32

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3_dcoa = self.Conv3_dc(x3)
        x3 = self.Conv3_sk(x3, x3_dcoa)
        # x3_odconv = self.Conv3_odconv(x3)
        # x3 = self.Conv3_sk(x3, x3_odconv)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x4_dcoa = self.Conv4_dc(x4)
        x4 = self.Conv4_sk(x4, x4_dcoa)
        # x4_odconv = self.Conv4_odconv(x4)
        # x4 = self.Conv4_sk(x4, x4_odconv)

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



# Main Model ---- the proposed OCE-Net

class OCENet_mine(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, downsize = 2, bilinear=True):
        super(OCENet_mine, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64//downsize)    # plain conv
        # self.Conv1_dc = conv_block_dcoa(ch_in=64//downsize, ch_out=64//downsize)    # Dynamic Complex Orientation Aware Convolution DCOA Conv
        self.Conv1_odconv = conv_block_odconv(ch_in=64 // downsize, ch_out=64 // downsize)  # Omni-Dimensional Dynamic Convolution ODConv
        self.Conv1_sk = SKConv(64 // downsize)    #  use SK Attention module to fuse the DCOA features (orientation features) and the plain features
        # self.Conv1_sk = nn.Conv2d(64//downsize, 64//downsize, kernel_size=1)   # the performance will drop if we use conv instead of SK_conv
        self.Fusion1 = conv_block(ch_in=(64 + 64) // downsize, ch_out=64 // downsize)


        self.Conv2 = conv_block(ch_in=64//downsize, ch_out=128//downsize)
        self.Conv2_dc = conv_block_dcoa(ch_in=128//downsize, ch_out=128//downsize)
        self.Conv2_odconv = conv_block_odconv(ch_in=128 // downsize, ch_out=128 // downsize)
        self.Conv2_sk = SKConv(128//downsize)
        # self.Conv2_sk = nn.Conv2d(128 // downsize, 128 // downsize, kernel_size=1)
        self.Fusion2 = conv_block(ch_in=(128 + 128) // downsize, ch_out=128 // downsize)

        self.Conv3 = conv_block(ch_in=128//downsize, ch_out=256//downsize)   #plain conv
        self.Conv3_dc = conv_block_dcoa(ch_in=256//downsize, ch_out=256//downsize)    # DCOA Conv
        self.Conv3_odconv = conv_block_odconv(ch_in=256 // downsize, ch_out=256 // downsize)  # ODConv
        self.Conv3_sk = SKConv(256 // downsize)
        # self.Conv3_sk = nn.Conv2d(256 // downsize, 256 // downsize, kernel_size=1)
        self.Fusion3 = conv_block(ch_in=(256 + 256) // downsize, ch_out=256 // downsize)

        self.Conv4 = conv_block(ch_in=256//downsize, ch_out=512//downsize)
        self.Conv4_dc = conv_block_dcoa(ch_in=512//downsize, ch_out=512//downsize)
        self.Conv4_odconv = conv_block_odconv(ch_in=512 // downsize, ch_out=512 // downsize)
        self.Conv4_sk = SKConv(512//downsize)
        self.Fusion4 = conv_block(ch_in=(512 + 512) // downsize, ch_out=512 //downsize)

        self.Up4 = up_conv(ch_in=512//downsize, ch_out=256//downsize)
        self.Up_conv4 = conv_block(ch_in=512//downsize, ch_out=256//downsize)

        self.Up3 = up_conv(ch_in=256//downsize, ch_out=128//downsize)
        self.Up3_dc = up_conv_dcoa(ch_in=256//downsize, ch_out=128//downsize)    # DCOA UP_CONV
        self.Up3_odconv = up_conv_odconv(ch_in=256 // downsize, ch_out=128 // downsize)  # ODConv UP_CONV
        self.Up3_sk = SKConv(256//downsize)
        # self.Up3_sk = nn.Conv2d(256 // downsize, 256 // downsize, kernel_size=1)

        self.Up_conv3 = conv_block_dcoa(ch_in=256//downsize, ch_out=128//downsize)

        self.Up2 = up_conv(ch_in=128//downsize, ch_out=64//downsize)
        self.Up2_dc = up_conv_dcoa(ch_in=128//downsize, ch_out=64//downsize)
        self.Up2_sk = SKConv(128//downsize)
        # self.Up2_sk = nn.Conv2d(128 // downsize, 128 // downsize, kernel_size=1)

        self.Up_conv2 = conv_block_dcoa(ch_in=128//downsize, ch_out=64//downsize)

        self.Conv_1x1 = nn.Conv2d(64//downsize, output_ch, kernel_size=1, stride=1, padding=0)

        self.conv_prob = nn.Conv2d(64 // downsize, 1, kernel_size=1, stride=1, padding=0)

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
        # x1_dcoa = self.Conv1_dc(x1) # 32,64,64
        # x1 = self.Conv1_sk(x1, x1_dcoa) # 1,64,64
        x1_odconv = self.Conv1_odconv(x1)  # 32,64,64
        x1 = self.Conv1_sk(x1, x1_odconv)  # 1,64,64

        x2 = self.Maxpool(x1)   # 32,32,32
        x2 = self.Conv2(x2)     # 64,32,32
        # x2_dcoa = self.Conv2_dc(x2) # 64,32,32
        # x2 = self.Conv2_sk(x2, x2_dcoa) # 64,32,32
        x2_odconv = self.Conv2_odconv(x2)  # 64,32,32
        x2 = self.Conv2_sk(x2, x2_odconv)  # 64,32,32

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        # x3_dcoa = self.Conv3_dc(x3)
        # x3 = self.Conv3_sk(x3, x3_dcoa)
        x3_odconv = self.Conv3_odconv(x3)
        x3 = self.Conv3_sk(x3, x3_odconv)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        # x4_dcoa = self.Conv4_dc(x4)
        # x4 = self.Conv4_sk(x4, x4_dcoa)
        x4_odconv = self.Conv4_odconv(x4)
        x4 = self.Conv4_sk(x4, x4_odconv)

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

