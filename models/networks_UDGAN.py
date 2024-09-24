import torch
import torch.nn as nn
from torch.autograd import Variable


class UDGAN_Generator_Twin(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=64, growth_rate=32, bn_size=4, drop_rate=0):
        super(UDGAN_Generator_Twin, self).__init__()
        self.input_A = UDGAN_Down(input_nc, ngf=ngf, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        self.input_B = UDGAN_Down(input_nc, ngf=ngf, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        self.output_F = UDGAN_Up(output_nc, ngf=ngf, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)

    def forward(self, img_A, img_B):
        xa = self.input_A(img_A)
        xb = self.input_B(img_B)
        xf = []
        for i in range(len(xa)):
            # xb[i] = xb[i].to(xa[0].device)
            # xa[i] = xa[i].to(xa[0].device)
            xf.append(torch.cat((xa[i], xb[i]), 1))
        out = self.output_F(xf)

        return out


class UDGAN_Generator_No_Connection(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=64, growth_rate=32, bn_size=4, drop_rate=0):
        super(UDGAN_Generator_No_Connection, self).__init__()
        self.input_A = UDGAN_Down(input_nc, ngf=ngf, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        self.input_B = UDGAN_Down(input_nc, ngf=ngf, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        self.output_F = UDGAN_Up_No_Connection(output_nc, ngf=ngf, growth_rate=growth_rate, bn_size=bn_size,
                                               drop_rate=drop_rate)

    def forward(self, img_A, img_B):
        xa = self.input_A(img_A)
        xb = self.input_B(img_B)
        xf = torch.cat((xa[6], xb[6]), 1)
        out = self.output_F(xf)

        return out


class UDGAN_Generator_No_Denseblocks(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=64):
        super(UDGAN_Generator_No_Denseblocks, self).__init__()
        self.input_A = UDGAN_Down_No_Denseblocks(input_nc, ngf=ngf)
        self.input_B = UDGAN_Down_No_Denseblocks(input_nc, ngf=ngf)
        self.output_F = UDGAN_Up_No_Denseblocks(output_nc, ngf=ngf)

    def forward(self, img_A, img_B):
        xa = self.input_A(img_A)
        xb = self.input_B(img_B)
        xf = []
        for i in range(len(xa)):
            xf.append(torch.cat((xa[i], xb[i]), 1))
        out = self.output_F(xf)

        return out


class UDGAN_Generator_Single(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=64, growth_rate=32, bn_size=4, drop_rate=0):
        super(UDGAN_Generator_Single, self).__init__()
        self.input = UDGAN_Down(input_nc, ngf=ngf, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        self.output = UDGAN_Up(output_nc, ngf=ngf, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate,
                               twin=False)

    def forward(self, img):
        x = self.input(img)
        out = self.output(x)

        return out


class UDGAN_Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, growth_rate=16, bn_size=4, drop_rate=0):
        super(UDGAN_Discriminator, self).__init__()
        max_nc = 8 * ndf
        num_dense = 3

        sequence = [
            UDGAN_Down_Block(input_nc, ndf, num_dense=num_dense, outer=True,
                             max_nc=max_nc, growth_rate=growth_rate * 1, bn_size=bn_size, drop_rate=drop_rate),
            UDGAN_Down_Block((ndf * 1 + growth_rate * 1 * num_dense * 1) // 2, ndf * 2, num_dense=num_dense * 2,
                             max_nc=max_nc, growth_rate=growth_rate * 2, bn_size=bn_size, drop_rate=drop_rate),
            UDGAN_Down_Block((ndf * 2 + growth_rate * 2 * num_dense * 2) // 2, ndf * 4, num_dense=num_dense * 4,
                             max_nc=max_nc, growth_rate=growth_rate * 4, bn_size=bn_size, drop_rate=drop_rate),
            nn.Conv2d((ndf * 4 + growth_rate * 4 * num_dense * 4) // 2, 1, kernel_size=3, stride=1, padding=1)
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        """Standard forward."""
        return self.model(x)


class UDGAN_Down(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, ngf=64, growth_rate=32, bn_size=4, drop_rate=0):
        super(UDGAN_Down, self).__init__()
        max_nc = 8 * ngf
        num_dense = 3
        self.down1 = UDGAN_Down_Block(input_nc, ngf * 1,
                                      num_dense=num_dense * 1, growth_rate=growth_rate * 1,
                                      bn_size=bn_size, drop_rate=drop_rate, max_nc=max_nc, outer=True)
        self.down2 = UDGAN_Down_Block((ngf * 1 + growth_rate * 1 * num_dense * 1) // 2, ngf * 2,
                                      num_dense=num_dense * 1, growth_rate=growth_rate * 1,
                                      bn_size=bn_size, drop_rate=drop_rate, max_nc=max_nc)
        self.down3 = UDGAN_Down_Block((ngf * 2 + growth_rate * 1 * num_dense * 1) // 2, ngf * 4,
                                      num_dense=num_dense * 2, growth_rate=growth_rate * 2,
                                      bn_size=bn_size, drop_rate=drop_rate, max_nc=max_nc)
        self.down4 = UDGAN_Down_Block((ngf * 4 + growth_rate * 2 * num_dense * 2) // 2, ngf * 4,
                                      num_dense=num_dense * 2, growth_rate=growth_rate * 2,
                                      bn_size=bn_size, drop_rate=drop_rate, max_nc=max_nc)
        self.down5 = UDGAN_Down_Block((ngf * 4 + growth_rate * 2 * num_dense * 2) // 2, ngf * 8,
                                      num_dense=num_dense * 4, growth_rate=growth_rate * 4,
                                      bn_size=bn_size, drop_rate=drop_rate, max_nc=max_nc)
        self.down6 = UDGAN_Down_Block((ngf * 8 + growth_rate * 4 * num_dense * 4) // 2, ngf * 8,
                                      num_dense=num_dense * 4, growth_rate=growth_rate * 4,
                                      bn_size=bn_size, drop_rate=drop_rate, max_nc=max_nc)
        self.down7 = nn.Sequential(
            nn.BatchNorm2d((ngf * 8 + growth_rate * 4 * num_dense * 4) // 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d((ngf * 8 + growth_rate * 4 * num_dense * 4) // 2, ngf * 8, kernel_size=3, padding=1, stride=2)
        )

    def forward(self, x):
        """Standard forward"""
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)

        return x1, x2, x3, x4, x5, x6, x7


class UDGAN_Down_No_Denseblocks(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, ngf=64):
        super(UDGAN_Down_No_Denseblocks, self).__init__()

        self.down1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, stride=2, padding=1, kernel_size=3),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, stride=2, padding=1, kernel_size=3),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, stride=2, padding=1, kernel_size=3),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 8, stride=2, padding=1, kernel_size=3),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, stride=2, padding=1, kernel_size=3),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True)
        )
        self.down6 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, stride=2, padding=1, kernel_size=3),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True)
        )
        self.down7 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, stride=2, padding=1, kernel_size=3)
        )

    def forward(self, x):
        """Standard forward"""
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)

        return x1, x2, x3, x4, x5, x6, x7


class UDGAN_Up(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, output_nc, ngf=64, growth_rate=32, bn_size=4, drop_rate=0, twin=True):
        super(UDGAN_Up, self).__init__()
        max_nc = 8 * ngf
        num_dense = 3
        if twin:
            multi_factor = 1.5
            up7_factor = 2
        else:
            multi_factor = 1
            up7_factor = 1
        self.up7 = UDGAN_Up_Block(ngf * 8 * up7_factor, ngf * 8,
                                  num_dense=num_dense * 4,
                                  max_nc=max_nc, growth_rate=growth_rate * 4, bn_size=bn_size, drop_rate=drop_rate)
        self.up6 = UDGAN_Up_Block(int((ngf * 8 + growth_rate * 4 * num_dense * 4) * multi_factor), ngf * 8,
                                  num_dense=num_dense * 4, growth_rate=growth_rate * 4,
                                  bn_size=bn_size, drop_rate=drop_rate, max_nc=max_nc)
        self.up5 = UDGAN_Up_Block(int((ngf * 8 + growth_rate * 4 * num_dense * 4) * multi_factor), ngf * 4,
                                  num_dense=num_dense * 2, growth_rate=growth_rate * 2,
                                  bn_size=bn_size, drop_rate=drop_rate, max_nc=max_nc)
        self.up4 = UDGAN_Up_Block(int((ngf * 4 + growth_rate * 2 * num_dense * 2) * multi_factor), ngf * 4,
                                  num_dense=num_dense * 2, growth_rate=growth_rate * 2,
                                  bn_size=bn_size, drop_rate=drop_rate, max_nc=max_nc)
        self.up3 = UDGAN_Up_Block(int((ngf * 4 + growth_rate * 2 * num_dense * 2) * multi_factor), ngf * 2,
                                  num_dense=num_dense * 1, growth_rate=growth_rate * 1,
                                  bn_size=bn_size, drop_rate=drop_rate, max_nc=max_nc)
        self.up2 = UDGAN_Up_Block(int((ngf * 2 + growth_rate * 1 * num_dense * 1) * multi_factor), ngf * 1,
                                  num_dense=num_dense * 1, growth_rate=growth_rate,
                                  bn_size=bn_size, drop_rate=drop_rate, max_nc=max_nc)
        self.up1 = nn.Sequential(
            nn.BatchNorm2d(int((ngf + growth_rate * 1 * num_dense) * multi_factor)),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(int((ngf + growth_rate * 1 * num_dense) * multi_factor), output_nc,
                               kernel_size=3, padding=1, output_padding=1, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        """Standard forward"""
        y7 = self.up7(x[6])
        y6 = self.up6(torch.cat((x[5], y7), 1))
        y5 = self.up5(torch.cat((x[4], y6), 1))
        y4 = self.up4(torch.cat((x[3], y5), 1))
        y3 = self.up3(torch.cat((x[2], y4), 1))
        y2 = self.up2(torch.cat((x[1], y3), 1))
        y1 = self.up1(torch.cat((x[0], y2), 1))

        return y1


class UDGAN_Up_No_Connection(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, output_nc, ngf=64, growth_rate=32, bn_size=4, drop_rate=0, twin=True):
        super(UDGAN_Up_No_Connection, self).__init__()
        max_nc = 8 * ngf
        num_dense = 3
        if twin:
            multi_factor = 1.5
            up7_factor = 2
        else:
            multi_factor = 1
            up7_factor = 1
        self.up7 = UDGAN_Up_Block(ngf * 8 * up7_factor, ngf * 8,
                                  num_dense=num_dense * 4,
                                  max_nc=max_nc, growth_rate=growth_rate * 4, bn_size=bn_size, drop_rate=drop_rate)
        self.up6 = UDGAN_Up_Block(ngf * 8 * 2, ngf * 8,
                                  num_dense=num_dense * 4, growth_rate=growth_rate * 4,
                                  bn_size=bn_size, drop_rate=drop_rate, max_nc=max_nc)
        self.up5 = UDGAN_Up_Block(ngf * 8 * 2, ngf * 4,
                                  num_dense=num_dense * 2, growth_rate=growth_rate * 2,
                                  bn_size=bn_size, drop_rate=drop_rate, max_nc=max_nc)
        self.up4 = UDGAN_Up_Block(ngf * 5, ngf * 4,
                                  num_dense=num_dense * 2, growth_rate=growth_rate * 2,
                                  bn_size=bn_size, drop_rate=drop_rate, max_nc=max_nc)
        self.up3 = UDGAN_Up_Block(ngf * 5, ngf * 2,
                                  num_dense=num_dense * 1, growth_rate=growth_rate * 1,
                                  bn_size=bn_size, drop_rate=drop_rate, max_nc=max_nc)
        self.up2 = UDGAN_Up_Block(112, ngf * 1,
                                  num_dense=num_dense * 1, growth_rate=growth_rate,
                                  bn_size=bn_size, drop_rate=drop_rate, max_nc=max_nc)
        self.up1 = nn.Sequential(
            nn.BatchNorm2d(80),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(80, output_nc,
                               kernel_size=3, padding=1, output_padding=1, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        """Standard forward"""
        y7 = self.up7(x)
        # print(y7.shape)
        y6 = self.up6(y7)
        # print(y6.shape)
        y5 = self.up5(y6)
        # print(y5.shape)
        y4 = self.up4(y5)
        # print(y4.shape)
        y3 = self.up3(y4)
        # print(y3.shape)
        y2 = self.up2(y3)
        # print(y2.shape)
        y1 = self.up1(y2)
        # print(y1.shape)

        return y1


class UDGAN_Up_No_Denseblocks(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, output_nc, ngf=64):
        super(UDGAN_Up_No_Denseblocks, self).__init__()

        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=3, padding=1, output_padding=1, stride=2),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True)
        )
        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * 3, ngf * 8, kernel_size=3, padding=1, output_padding=1, stride=2),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True)
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * 3, ngf * 8, kernel_size=3, padding=1, output_padding=1, stride=2),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * 3, ngf * 4, kernel_size=3, padding=1, output_padding=1, stride=2),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4 * 3, ngf * 2, kernel_size=3, padding=1, output_padding=1, stride=2),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2 * 3, ngf, kernel_size=3, padding=1, output_padding=1, stride=2),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, True)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 3, output_nc, kernel_size=3, padding=1, output_padding=1, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        """Standard forward"""
        y7 = self.up7(x[6])
        y6 = self.up6(torch.cat((x[5], y7), 1))
        y5 = self.up5(torch.cat((x[4], y6), 1))
        y4 = self.up4(torch.cat((x[3], y5), 1))
        y3 = self.up3(torch.cat((x[2], y4), 1))
        y2 = self.up2(torch.cat((x[1], y3), 1))
        y1 = self.up1(torch.cat((x[0], y2), 1))

        return y1


class UDGAN_Down_Block(nn.Module):
    def __init__(self, input_down, output_down, max_nc, num_dense, growth_rate, bn_size, drop_rate=0, outer=False):
        super(UDGAN_Down_Block, self).__init__()
        if output_down >= max_nc:
            output_down = max_nc

        if outer:
            self.convlayer = nn.Sequential(
                nn.Conv2d(input_down, output_down, kernel_size=3, padding=1, stride=2)
            )
        else:
            self.convlayer = nn.Sequential(
                nn.BatchNorm2d(input_down),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(input_down, output_down, kernel_size=3, padding=1, stride=2)
            )
        self.denselayer = Dense_Block(num_dense, output_down, growth_rate, bn_size, drop_rate)
        output_dense = output_down + growth_rate * num_dense
        self.transition = TransitionLayer(output_dense, output_dense // 2)

    def forward(self, x):
        x = self.convlayer(x)
        x = self.denselayer(x)
        x = self.transition(x)
        return x


class UDGAN_Up_Block(nn.Module):
    def __init__(self, input_up, output_up, max_nc, num_dense, growth_rate, bn_size, drop_rate):
        super(UDGAN_Up_Block, self).__init__()
        if output_up >= max_nc:
            output_up = max_nc

        self.convlayer = nn.Sequential(
            nn.BatchNorm2d(input_up),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(input_up, output_up, kernel_size=3, padding=1, output_padding=1, stride=2)
        )
        # ).to('cuda:0')
        self.denselayer = Dense_Block(num_dense, output_up, growth_rate, bn_size, drop_rate)
        output_dense = output_up + growth_rate * num_dense
        self.transition = TransitionLayer(output_dense, output_dense // 2)

    def forward(self, x):
        x = self.convlayer(x)
        x = self.denselayer(x)
        x = self.transition(x)
        return x


class DenseLayer(nn.Module):
    def __init__(self, inplace, growth_rate, bn_size, drop_rate=0):
        super(DenseLayer, self).__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.Sequential(
            nn.BatchNorm2d(inplace),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=inplace, out_channels=bn_size * growth_rate,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=bn_size * growth_rate, out_channels=growth_rate,
                      kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        y = self.dense_layer(x)
        if self.drop_rate > 0:
            y = self.dropout(y)
        return torch.cat([x, y], 1)


class Dense_Block(nn.Module):
    def __init__(self, num_layers, inplances, growth_rate=32, bn_size=4, drop_rate=0):
        super(Dense_Block, self).__init__()
        dense_layers = []
        self.output_nc = 0

        for i in range(num_layers):
            dense_layers.append(DenseLayer(inplances + i * growth_rate, growth_rate, bn_size, drop_rate))

        self.dense_layers = nn.Sequential(*dense_layers)

    def forward(self, x):
        y_dense = self.dense_layers(x)
        self.output_nc = int(y_dense.shape[1])
        return y_dense


class TransitionLayer(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(TransitionLayer, self).__init__()
        self.transition_layer = nn.Sequential(
            nn.BatchNorm2d(input_nc),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_nc, out_channels=output_nc, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        x = self.transition_layer(x)
        return x


# net = UDGAN_Generator_Twin(3, 3)
# sampledataA = Variable(torch.randn(2, 3, 128, 128))
# sampledataB = Variable(torch.randn(2, 3, 128, 128))
# output = net(sampledataA, sampledataB)
# print(output.shape)
# print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))

# net = UDGAN_Generator_No_Denseblocks(3, 3)
# sampledataA = torch.randn(2, 3, 128, 128)
# # print(net)
# sampledataB = torch.randn(2, 3, 128, 128)
# output = net(sampledataA, sampledataB)
# print(output.shape)
# print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))

# net = UDGAN_Generator_single(3, 3)
# sampledataA = Variable(torch.randn(2, 3, 128, 128))
# output = net(sampledataA)
# print(output.shape)
# print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))


# net = UDGAN_Discriminator(3)
# sampledataA = Variable(torch.randn(2, 3, 128, 128))
# output = net(sampledataA)
# print(output.shape)
# print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))
