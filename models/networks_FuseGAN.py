import torch
import torch.nn as nn
from torch.autograd import Variable


class FuseGAN_Generator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc):
        super(FuseGAN_Generator, self).__init__()
        self.branchA = branch_conv(input_nc)
        self.branchB = branch_conv(input_nc)
        sequence = [
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256, eps=0.00001, momentum=0.1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128, eps=0.00001, momentum=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=output_nc, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()
        ]
        self.branchOut = nn.Sequential(*sequence)

    def forward(self, x, y):
        x = self.branchA(x)
        y = self.branchB(y)
        out = self.branchOut(torch.cat((x, y), 1))
        return out


class FuseGAN_Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(FuseGAN_Discriminator, self).__init__()
        sequence = [
            nn.Conv2d(in_channels=input_nc, out_channels=64, kernel_size=4, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(128, eps=0.00001, momentum=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(256, eps=0.00001, momentum=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(512, eps=0.00001, momentum=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(512, eps=0.00001, momentum=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(512, eps=0.00001, momentum=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(512, eps=0.00001, momentum=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=2),
            nn.Sigmoid()
        ]
        self.net = nn.Sequential(*sequence)

    def forward(self, x):
        return self.net(x)


class branch_conv(nn.Module):
    def __init__(self, input_nc):
        super(branch_conv, self).__init__()
        sequence1to3 = [
            nn.Conv2d(in_channels=input_nc, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ]
        sequence4to14 = []
        conv_conv_block = [
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=0.00001, momentum=0.1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=0.00001, momentum=0.1)
        ]
        for i in range(10):
            sequence4to14 += conv_conv_block

        self.conv1to3 = nn.Sequential(*sequence1to3)
        self.conv4to14 = nn.Sequential(*sequence4to14)

    def forward(self, x):
        x = self.conv1to3(x)
        return self.conv4to14(x)


# net = FuseGAN_Generator()
# sampledataA = Variable(torch.randn(2, 3, 128, 128))
# sampledataB = Variable(torch.randn(2, 3, 128, 128))
# output = net(sampledataA, sampledataB)
# print(output.shape)
# print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))


