import torch

import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, out_ch):
        super(Conv, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, 1, 1), nn.LeakyReLU(negative_slope=1e-2, inplace=True))

    def forward(self, x):
        return self.body(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return x


class SpatialAttentionBlock(nn.Module):
    def __init__(self):
        super(SpatialAttentionBlock, self).__init__()
        self.att1 = SpatialAttention()
        self.att2 = SpatialAttention()

    def forward(self, x1, x2):
        EPSILON = 1e-10
        att1 = self.att1(x1)
        att2 = self.att2(x2)
        mask1 = torch.exp(att1) / (torch.exp(att2) + torch.exp(att2) + EPSILON)
        mask2 = torch.exp(att2) / (torch.exp(att1) + torch.exp(att2) + EPSILON)
        x1_a = mask1 * x1
        x2_a = mask2 * x2
        return x1_a, x2_a


class ChannelAttention(nn.Module):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.avg(x)


class ChannelAttentionBlock(nn.Module):
    def __init__(self):
        super(ChannelAttentionBlock, self).__init__()
        self.ca1 = nn.AdaptiveAvgPool2d(1)
        self.ca2 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x1, x2):
        EPSILON = 1e-10
        ca1 = self.ca1(x1)
        ca2 = self.ca1(x2)
        mask1 = torch.exp(ca1) / (torch.exp(ca2) + torch.exp(ca1) + EPSILON)
        mask2 = torch.exp(ca2) / (torch.exp(ca1) + torch.exp(ca2) + EPSILON)
        x1_a = mask1 * x1
        x2_a = mask2 * x2
        return x1_a, x2_a


class attention(nn.Module):
    def __init__(self, out_ch):
        super(attention, self).__init__()
        self.conv1 = Conv(out_ch=out_ch)
        self.conv2 = Conv(out_ch=out_ch)
        self.att_ch = ChannelAttentionBlock()
        self.att_sp = SpatialAttentionBlock()

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv1(x2)
        x1_c, x2_c = self.att_ch(x1, x2)
        x1_s, x2_s = self.att_sp(x1_c, x2_c)
        x1_out = self.conv2(x1_s)
        x2_out = self.conv2(x2_s)
        return x1_out, x2_out, x1, x2, x1_s, x2_s


class Resblock(nn.Module):
    def __init__(self, out_ch):
        super(Resblock, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, 1, 1), nn.LeakyReLU(negative_slope=1e-2, inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, 1, 1))

    def forward(self, x1, x2):
        x1_out = self.body(x1)
        x2_out = self.body(x2)
        return x1_out, x2_out


class AF(nn.Module):
    def __init__(self, args):
        super(AF, self).__init__()
        in_ch = args.in_ch
        out_ch = args.out_ch
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.LeakyReLU(negative_slope=1e-2, inplace=True))
        self.rb1 = Resblock(out_ch)
        self.rb2 = Resblock(out_ch)
        self.rb3 = Resblock(out_ch)
        self.sc4 = attention(out_ch)
        self.fusion = nn.Conv2d(out_ch * 2, out_ch, 1, 1, 0)
        EX3 = [Conv(out_ch=out_ch) for i in range(4)]
        self.EX3 = nn.Sequential(*EX3)
        # conv
        self.restruction = nn.Conv2d(out_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x1, x2):
        F1 = self.conv1(x1)
        F2 = self.conv1(x2)
        F1, F2 = self.rb1(F1, F2)
        F1, F2 = self.rb2(F1, F2)
        F1, F2 = self.rb3(F1, F2)
        F1, F2, F1_ori, F2_ori, F1_att, F2_att = self.sc4(F1, F2)
        fusion = self.fusion(torch.cat([F1, F2], dim=1))
        fusion = self.EX3(fusion)
        output = self.restruction(fusion)
        output = self.sig(output)
        return output, F1_ori, F2_ori, F1_att, F2_att
