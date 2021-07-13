import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import os

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        # self.gate_activation = gate_activation
        self.gate_c = nn.Sequential()
        self.gate_c.add_module('flatten', Flatten())
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]

        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module('gate_c_fc_%d' % i, nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_bn_%d' % (i + 1), nn.BatchNorm1d(gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_relu_%d' % (i + 1), nn.ReLU())
        self.gate_c.add_module('gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d(in_tensor, in_tensor.size(2), stride=in_tensor.size(2))
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SpatialGate(nn.Module):
    def __init__(self, gate_channel):
        super(SpatialGate, self).__init__()
        input_channel = gate_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, 3, stride=1, padding=1),
            nn.BatchNorm2d(input_channel),
            # nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, 3, stride=1, padding=1),
            nn.BatchNorm2d(input_channel),
            # nn.ReLU()
        )
        self.deconv1 = nn.ConvTranspose2d(input_channel, input_channel, 3, stride=1, padding=1)

    def forward(self, in_tensor):
        c1 = self.conv1(in_tensor)
        c2 = self.conv2(c1)
        de_c2 = self.deconv1(c2)
        de_c1 = de_c2 + c1
        de_c1_s = self.deconv1(de_c1)
        de_in = de_c1_s + in_tensor
        final_c_2 = self.conv1(c2)
        final_c_1 = self.conv1(de_c1)
        final_c_in = self.conv1(de_in)
        final_c_2 = self.deconv1(final_c_2)
        final_c_1 = final_c_1 + final_c_2
        final_c_1 = self.deconv1(final_c_1)
        final_c_in = final_c_in + final_c_1
        return final_c_in

class Att(nn.Module):
    def __init__(self, gate_channel):
        super(Att, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)

    def forward(self, in_tensor):
        att = 1 + F.sigmoid(torch.mul(self.channel_att(in_tensor), self.spatial_att(in_tensor)))
        return in_tensor + torch.mul(att, in_tensor) 
        
class UAtt(nn.Module):
    def __init__(self, gate_channel):
        super(Att, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)

    def forward(self, in_tensor1, in_tensor2):
        att = 1 + F.sigmoid(torch.mul(self.channel_att(in_tensor1), self.spatial_att(in_tensor2)))
        return in_tensor1 + torch.mul(att, in_tensor2) 
        
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()

        self.inc = DoubleConv(3, 64)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Att1 = Att(64)
        self.Conv1 = conv_block(ch_in=64, ch_out=128)

        self.Att2 = Att(128)
        self.Conv2 = conv_block(ch_in=128, ch_out=256)

        self.Conv3 = conv_block(ch_in=256, ch_out=512)

        self.Conv4 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.UAtt3 = Att(128)
        
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.UAtt2 = Att(64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.inc(x)

        x2 = self.Maxpool(x1)
        x2 = self.Att1(x2)
        x2 = self.Conv1(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Att2(x3)
        x3 = self.Conv2(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv3(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv4(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.Utt2(d3, x2)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.Utt1(d2, x1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = F.sigmoid(d1)

        return d1

def unet():
    model = U_Net()
    return model
