import torch
import torch.nn as nn
from torch.autograd import Variable


class main_branch(nn.Module):
    def __init__(self, input_nc):
        super(main_branch, self).__init__()
        sequence1 = [
            nn.Conv2d(in_channels=input_nc, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ]
        sequence2 = [
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ]
        sequence3 = [
            nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ]
        sequence4 = [
            nn.Conv2d(in_channels=448, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ]

        self.conv1 = nn.Sequential(*sequence1)
        self.conv2 = nn.Sequential(*sequence2)
        self.conv3 = nn.Sequential(*sequence3)
        self.conv4 = nn.Sequential(*sequence4)

    def forward(self, x):
        y1 = self.conv1(x)
        x1 = y1
        y2 = self.conv2(x1)
        x2 = torch.cat([y1, y2], 1)
        y3 = self.conv3(x2)
        x3 = torch.cat([y1, y2, y3], 1)
        y4 = self.conv4(x3)
        x4 = torch.cat([y1, y2, y3, y4], 1)

        return x1, x2, x3, x4


class sub_branch_1(nn.Module):
    def __init__(self):
        super(sub_branch_1, self).__init__()
        sequence1 = [
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ]
        sequence2 = [
            nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ]
        sequence3 = [
            nn.Conv2d(in_channels=448, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ]

        self.conv1 = nn.Sequential(*sequence1)
        self.conv2 = nn.Sequential(*sequence2)
        self.conv3 = nn.Sequential(*sequence3)

    def forward(self, x):
        y1 = self.conv1(x)
        x1 = torch.cat([x, y1], 1)
        y2 = self.conv2(x1)
        x2 = torch.cat([x, y1, y2], 1)
        y3 = self.conv3(x2)
        x3 = torch.cat([x, y1, y2, y3], 1)

        return x1, x2, x3


class sub_branch_2(nn.Module):
    def __init__(self):
        super(sub_branch_2, self).__init__()
        sequence1 = [
            nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ]
        sequence2 = [
            nn.Conv2d(in_channels=448, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ]

        self.conv1 = nn.Sequential(*sequence1)
        self.conv2 = nn.Sequential(*sequence2)

    def forward(self, x):
        y1 = self.conv1(x)
        x1 = torch.cat([x, y1], 1)
        y2 = self.conv2(x1)
        x2 = torch.cat([x, y1, y2], 1)

        return x1, x2


class sub_branch_3(nn.Module):
    def __init__(self):
        super(sub_branch_3, self).__init__()
        sequence1 = [
            nn.Conv2d(in_channels=448, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ]

        self.conv1 = nn.Sequential(*sequence1)

    def forward(self, x):
        y1 = self.conv1(x)
        x1 = torch.cat([x, y1], 1)

        return x1


class fusion_layer(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(fusion_layer, self).__init__()
        conv = [
            nn.Conv2d(in_channels=input_nc, out_channels=output_nc, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(output_nc),
            nn.ReLU()
        ]
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class output_layer(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(output_layer, self).__init__()
        conv = [
            nn.Conv2d(in_channels=input_nc, out_channels=output_nc, kernel_size=1, padding=0, stride=1),
            nn.Tanh()
        ]
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class generator(nn.Module):
    def __init__(self, input, output):
        super(generator, self).__init__()
        self.mainnet_1 = main_branch(input)
        self.mainnet_2 = main_branch(input)
        self.subnet_1 = sub_branch_1()
        self.subnet_2 = sub_branch_2()
        self.subnet_3 = sub_branch_3()
        self.fusion_s1 = fusion_layer(64 * 2, 64)
        self.fusion_s2 = fusion_layer(192 * 3, 192)
        self.fusion_s3 = fusion_layer(448 * 4, 448)
        self.out_m = output_layer(960 * 2, output)
        self.out_s1 = output_layer(960, output)
        self.out_s2 = output_layer(960, output)
        self.out_s3 = output_layer(960, output)
        self.out_all = output_layer(960 * 5, output)

    def forward(self, imgA, imgB):
        m_x1_A, m_x2_A, m_x3_A, m_x4_A = self.mainnet_1(imgA)
        # print(m_x1_A[0, 0, :, :].unsqueeze(0).unsqueeze(0).shape)
        m_x1_B, m_x2_B, m_x3_B, m_x4_B = self.mainnet_2(imgB)
        # o_m = self.out_m(torch.cat([m_x4_A, m_x4_B], 1))
        i_s1 = self.fusion_s1(torch.cat([m_x1_A, m_x1_B], 1))
        s1_x1, s1_x2, s1_x3 = self.subnet_1(i_s1)
        # o_s1 = self.out_s1(s1_x3)
        i_s2 = self.fusion_s2(torch.cat([m_x2_A, m_x2_B, s1_x1], 1))
        s2_x1, s2_x2 = self.subnet_2(i_s2)
        # o_s2 = self.out_s2(s2_x2)
        i_s3 = self.fusion_s3(torch.cat([m_x3_A, m_x3_B, s1_x2, s2_x1], 1))
        s3_x1 = self.subnet_3(i_s3)
        # o_s3 = self.out_s3(s3_x1)
        i_all = torch.cat([m_x4_A, m_x4_B, s1_x3, s2_x2, s3_x1], 1)
        o_all = self.out_all(i_all)
        # print(o_all.shape)

        # return o_m, o_s1, o_s2, o_s3, o_all
        # return o_all
        return m_x4_A[0, 160, :, :].unsqueeze(0).unsqueeze(0)


# net1 = main_branch(3)
# sampledataA = Variable(torch.randn(2, 3, 128, 128))
# output1, output2, output3, output4 = net1(sampledataA)
# print(output1.shape, output2.shape, output3.shape, output4.shape)
# net = generator(3, 3)
# sampledataA = Variable(torch.randn(2, 3, 128, 128))
# sampledataB = Variable(torch.randn(2, 3, 128, 128))
# o1, o2, o3, o4, o5 = net(sampledataA, sampledataB)
# print(o1.shape, o2.shape, o3.shape, o4.shape, o5.shape)
# print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))


