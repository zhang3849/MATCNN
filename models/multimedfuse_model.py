#!/usr/bin/env python
import torch
import torch.nn as nn
from . import networks
import numpy as np
from .base_model import BaseModel
from torch.autograd import Variable
import torch.nn.functional as F
from util.ssim import SSIM
from swin import buildswin
from torchvision.transforms import Resize
from kornia.filters import gaussian_blur2d


def functional_conv2d(im):
    # im = (im + 1) / 2 * 255
    sobel_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype='float32')  #
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    weight = torch.from_numpy(sobel_kernel).to(im.device)
    edge_detect = F.conv2d(im, weight, padding=1, stride=1)
    edge_detect.required_grad = False
    return edge_detect ** 2


def bilinear_interpolation_naive(src, dst_size):
    """
    双线性差值的naive实现
    :param src: 源图像
    :param dst_size: 目标图像大小H*W
    :return: 双线性差值后的图像
    """
    src_c, src_h, src_w = src.shape[-3], src.shape[-2], src.shape[-1]  # 原图像大小 H*W*C
    (dst_h, dst_w), dst_c = dst_size, src_c  # 目标图像大小H*W*C

    if src_h == dst_h and src_w == dst_w:  # 如果大小不变，直接返回copy
        return src

    scale_h = float(src_h) / dst_h  # 计算H方向缩放比
    scale_w = float(src_w) / dst_w  # 计算W方向缩放比
    dst = torch.zeros((src.shape[0], src.shape[1], dst_h, dst_w), dtype=src.dtype)  # 目标图像初始化
    for h_d in range(dst_h):  # 遍历目标图像H方向
        for w_d in range(dst_w):  # 遍历目标图像所有W方向
            h = scale_h * (h_d + 0.5) - 0.5  # 将目标图像H坐标映射到源图像上
            w = scale_w * (w_d + 0.5) - 0.5  # 将目标图像W坐标映射到源图像上
            h0 = int(np.floor(h))  # 最近4个点坐标h0
            w0 = int(np.floor(w))  # 最近4个点坐标w0
            h1 = min(h0 + 1, src_h - 1)  # h0 + 1就是h1，但是不能越界
            w1 = min(w0 + 1, src_w - 1)  # w0 + 1就是w1，但是不能越界
            r0 = (w1 - w) * src[..., h0, w0] + (w - w0) * src[..., h0, w1]  # 双线性差值R0
            r1 = (w1 - w) * src[..., h1, w0] + (w - w0) * src[..., h1, w1]  # 双线性插值R1
            p = (h1 - h) * r0 + (h - h0) * r1  # 双线性插值P
            dst[..., h_d, w_d] = p  # 插值结果放进目标像素点
    return dst


class Clar_Loss(nn.Module):
    def __init__(self, k_size=3):
        super(Clar_Loss, self).__init__()
        self.k_size = k_size

    def forward(self, TensorA, TensorB, TensorF):
        LA = functional_conv2d(TensorA)
        LB = functional_conv2d(TensorB)
        LAB = 0.5 * (LA + LB + abs(LA - LB))  # max(LA,LB),Lglo,LA LB LF全局特征损失
        LF = functional_conv2d(TensorF)
        diff = torch.nn.L1Loss()
        loss = 0.2 * diff(LAB, LF) + diff(LB, LF)
        return loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]  # TV(x)
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2)  # .pow(2,3,3) 2的3次方，取模3
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2)  # 与论文不一致？多了平方
        return self.tv_loss_weight * 2 * (h_tv[:, :, :h_x - 1, :w_x - 1] + w_tv[:, :, :h_x - 1, :w_x - 1])

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class MULTIMEDFUSEModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='multiscale3', dataset_mode='fusion')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'G_SWIN', 'G_Clarity', 'G_TV', 'G_SSIM', 'DI_real', 'DV_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'real_B', 'fake_F']  # 'real_F'
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'DI', 'DV']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netDI = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netDV = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.swin_model = buildswin.build_swin(opt.gpu_ids)
            self.swin_model.eval()
            for param in self.swin_model.parameters():
                param.requires_grad = False
            # self.resize_t = Resize([224,224])

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.Clarity = Clar_Loss().to(self.device)
            self.TVloss = TVLoss().to(self.device)
            # self.DetailLoss = Detail_Loss.to(self.device)
            self.SSIM = SSIM()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_DI = torch.optim.Adam(self.netDI.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_DV = torch.optim.Adam(self.netDV.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_DI)
            self.optimizers.append(self.optimizer_DV)
            # self.cat = torch.cat()

    def set_input(self, input):
        self.ABtoF = self.opt.direction == 'ABtoF'
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.real_M = input['M'].to(self.device)
        # self.real_F = input['F'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.ABtoF:
            self.fake1, self.fake2, self.fake3, self.fake_F = self.netG(self.real_A, self.real_B)  # G(AB)
            # self.fake_F = self.netG(torch.cat((self.real_A, self.real_B), 1))
        # A = bilinear_interpolation_naive(self.real_A.repeat(1,3,1,1), (224, 224))
        # B = bilinear_interpolation_naive(self.real_B.repeat(1,3,1,1), (224, 224))
        # Fusion = bilinear_interpolation_naive(self.fake_F.repeat(1,3,1,1), (224, 224))
        # A = self.real_A.repeat(1, 3, 1, 1)
        # B = self.real_B.repeat(1, 3, 1, 1)
        # Fusion = self.fake_F.repeat(1, 3, 1, 1)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake
        fake_ABF = self.fake_F
        pred_fakeI = self.netDI(fake_ABF.detach())
        pred_fakeV = self.netDV(fake_ABF.detach())
        self.loss_DI_fake = self.criterionGAN(pred_fakeI, False)
        self.loss_DV_fake = self.criterionGAN(pred_fakeV, False)

        # Real
        real_ABA = self.real_A
        real_ABB = self.real_B
        pred_realI = self.netDI(real_ABA)
        pred_realV = self.netDV(real_ABB)
        self.loss_DI_real = self.criterionGAN(pred_realI, True)
        self.loss_DV_real = self.criterionGAN(pred_realV, True)
        # combine loss and calculate gradients
        self.loss_DI = (self.loss_DI_fake + self.loss_DI_real) * 0.5
        self.loss_DV = (self.loss_DV_fake + self.loss_DV_real) * 0.5
        self.loss_D = self.loss_DI + self.loss_DV
        self.loss_D = self.loss_D * 0
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_ABF = self.fake_F
        pred_fakeI = self.netDI(fake_ABF)
        pred_fakeV = self.netDV(fake_ABF)
        mask = torch.div(self.real_M, 255)
        self.loss_GI_GAN = self.criterionGAN(pred_fakeI, True)  # 均方误差
        self.loss_GV_GAN = self.criterionGAN(pred_fakeV, True)
        self.loss_G_GAN = self.loss_GI_GAN + self.loss_GV_GAN
        self.loss_G_GAN = self.loss_G_GAN * 0
        # Second, G(A) = B
        # self.loss_G_L1A = self.criterionL1(self.fake_F, self.real_A)
        # self.loss_G_L1B = self.criterionL1(self.fake_F, self.real_B)
        # self.loss_G_L1 = (self.criterionL1(self.fake_F, self.real_A) + 0.1 * self.criterionL1(self.fake_F, 0.5 * (self.real_A + self.real_B + abs(self.real_A - self.real_B)))) * self.opt.lambda_L1

        self.loss_G_L1 = (self.criterionL1(torch.mul(1 - mask, self.fake_F),torch.mul(1 - mask, self.real_B)) + 100 * self.criterionL1(torch.mul(mask, self.fake_F), torch.mul(mask, self.real_A))) * 10
        # self.loss_G_L1 = 0.5 * (self.loss_G_L1A + self.loss_G_L1B) * self.opt.lambda_L1
        # self.loss_G_Detail = self.DetailLoss(self.fake_F, self.real_A) * 200

        # self.loss_G_L2 = (self.criterionL2(self.fake_F, self.real_A) + 0.3 * self.criterionL2(self.fake_F, self.real_B)) * 20

        self.loss_G_Clarity1 = self.Clarity(self.real_A, self.real_B, self.fake1)  # 100
        self.loss_G_Clarity2 = self.Clarity(self.real_A, self.real_B, self.fake2)
        self.loss_G_Clarity3 = self.Clarity(self.real_A, self.real_B, self.fake3)
        self.loss_G_ClarityF = self.Clarity(self.real_A, self.real_B, self.fake_F)
        self.loss_G_Clarity = (self.loss_G_Clarity1 + self.loss_G_Clarity2 + self.loss_G_Clarity3 + self.loss_G_ClarityF) * 0.25 * 0
        self.loss_G_TV1 = (0.2 * self.criterionL1(self.TVloss(self.fake1), self.TVloss(self.real_A)) + self.criterionL1(self.TVloss(self.fake1), self.TVloss(self.real_B)))
        self.loss_G_TV2 = (0.2 * self.criterionL1(self.TVloss(self.fake2), self.TVloss(self.real_A)) + self.criterionL1(self.TVloss(self.fake2), self.TVloss(self.real_B)))
        self.loss_G_TV3 = (0.2 * self.criterionL1(self.TVloss(self.fake3), self.TVloss(self.real_A)) + self.criterionL1(self.TVloss(self.fake3), self.TVloss(self.real_B)))
        self.loss_G_TVF = (0.2 * self.criterionL1(self.TVloss(self.fake_F), self.TVloss(self.real_A)) + self.criterionL1(self.TVloss(self.fake_F), self.TVloss(self.real_B)))
        self.loss_G_TV = (self.loss_G_TV1 + self.loss_G_TV2 + self.loss_G_TV3 + self.loss_G_TVF) * 0.25 * 0
        self.loss_G_SSIMB = 1 - self.SSIM(self.fake_F, self.real_B)
        self.loss_G_SSIMA = 1 - self.SSIM(self.fake_F, self.real_A)
        # self.loss_G_SSIM = (0.1 * self.loss_G_SSIMA + self.loss_G_SSIMB) * 10
        self.loss_G_SSIM = (self.loss_G_SSIMA + self.loss_G_SSIMB) * 0.5 * 5

        A = nn.ZeroPad2d(padding=(48, 48, 48, 48))(self.real_A.repeat(1, 3, 1, 1))
        B = nn.ZeroPad2d(padding=(48, 48, 48, 48))(self.real_B.repeat(1, 3, 1, 1))
        Fusion1 = nn.ZeroPad2d(padding=(48, 48, 48, 48))(self.fake_F.repeat(1, 3, 1, 1))
        Fusion2 = nn.ZeroPad2d(padding=(48, 48, 48, 48))(self.fake_F.repeat(1, 3, 1, 1))
        Fusion3 = nn.ZeroPad2d(padding=(48, 48, 48, 48))(self.fake_F.repeat(1, 3, 1, 1))
        Fusion = nn.ZeroPad2d(padding=(48, 48, 48, 48))(self.fake_F.repeat(1, 3, 1, 1))
        self.feature_A = self.swin_model(A)
        self.feature_B = self.swin_model(B)
        self.feature1 = self.swin_model(Fusion1)
        self.feature2 = self.swin_model(Fusion2)
        self.feature3 = self.swin_model(Fusion3)
        self.feature_F = self.swin_model(Fusion)
        # self.loss_G_SWINa = [torch.mean(torch.abs(self.feature_F[i] - self.feature_A[i])) for i in range(4)] + \
        #                     [torch.mean(torch.abs(self.feature_F[i] - self.feature_B[i])) * 1.5 for i in range(4)]
        self.loss_G_SWIN1 = [(0.2 * self.criterionL1(self.feature1[i], 0.5 * (self.feature_A[i] + self.feature_B[i] + torch.abs(self.feature_A[i] - self.feature_B[i]))) + self.criterionL1(self.feature1[i], self.feature_A[i])) for i in range(4)]
        self.loss_G_SWIN2 = [(0.2 * self.criterionL1(self.feature2[i], 0.5 * (self.feature_A[i] + self.feature_B[i] + torch.abs(self.feature_A[i] - self.feature_B[i]))) + self.criterionL1(self.feature1[i], self.feature_A[i])) for i in range(4)]
        self.loss_G_SWIN3 = [(0.2 * self.criterionL1(self.feature3[i], 0.5 * (self.feature_A[i] + self.feature_B[i] + torch.abs(self.feature_A[i] - self.feature_B[i]))) + self.criterionL1(self.feature1[i], self.feature_A[i])) for i in range(4)]
        self.loss_G_SWINF = [(0.2 * self.criterionL1(self.feature_F[i], 0.5 * (self.feature_A[i] + self.feature_B[i] + torch.abs(self.feature_A[i] - self.feature_B[i]))) + self.criterionL1(self.feature1[i], self.feature_A[i])) for i in range(4)]
        # self.loss_G_SWINa = [torch.mean(torch.abs(self.feature_F[i] - (self.feature_A[i] ** 2 + self.feature_B[i] ** 2) / (torch.abs(self.feature_A[i]) + torch.abs(self.feature_B[i])))) for i in range(4)]
        self.loss_G_SWIN = (sum(self.loss_G_SWIN1) + sum(self.loss_G_SWIN2) + sum(self.loss_G_SWIN3) + sum(self.loss_G_SWINF)) * 0.25 * 10

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_Clarity + self.loss_G_SSIM + self.loss_G_TV + self.loss_G_SWIN
        # self.loss_G = self.loss_G_GAN + self.loss_G_L1
        # self.loss_G = self.loss_G_GAN
        # self.loss_G = self.loss_G_GAN + self.loss_G_Clarity
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netDI, True)  # enable backprop for D
        self.set_requires_grad(self.netDV, True)  # enable backprop for D
        self.optimizer_DI.zero_grad()  # set D's gradients to zero
        self.optimizer_DV.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_DI.step()  # update D's weights
        self.optimizer_DV.step()  # update D's weights

        # update G
        self.set_requires_grad(self.netDI, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netDV, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
