from skimage import io, color
import numpy as np
import torch
import cv2
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io, transform, filters, img_as_float, img_as_ubyte


def SML(tensor):
    H, W = tensor.shape
    NML = torch.zeros([H, W])

    for i in range(H):
        for j in range(W):
            pix_up = i - 1
            pix_bottom = i + 1
            pix_left = j - 1
            pix_right = j + 1

            if pix_up < 0:
                pix_up = 0
            if pix_bottom > H - 1:
                pix_bottom = H - 1
            if pix_left < 0:
                pix_left = 0
            if pix_right > W - 1:
                pix_right = W - 1

            NML[i][j] = \
                abs(tensor[i][j] - tensor[pix_up][j]) + abs(tensor[i][j] - tensor[pix_bottom][j]) + \
                abs(tensor[i][j] - tensor[i][pix_left]) + abs(tensor[i][j] - tensor[i][pix_right]) + \
                0.707 * abs(tensor[i][j] - tensor[pix_up][pix_left]) + 0.707 * abs(
                    tensor[i][j] - tensor[pix_bottom][pix_right]) + \
                0.707 * abs(tensor[i][j] - tensor[pix_up][pix_right]) + 0.707 * abs(
                    tensor[i][j] - tensor[pix_bottom][pix_left])

    return NML


def NSML(tensor):
    NML = SML(tensor)
    H_n, W_n = NML.shape
    NSML_in = torch.zeros([H_n + 2, W_n + 2])
    NSML_in[1:H_n + 1, 1:W_n + 1] = NML[0:H_n, 0:W_n]
    NSML_o = torch.zeros([H_n + 2, W_n + 2])
    window = torch.zeros([3, 3])
    data = 1 / 16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    omiga = torch.tensor(data)

    for m in range(1, H_n + 1):
        for n in range(1, W_n + 1):
            I_up = m - 1
            I_bottom = m + 1
            I_left = n - 1
            I_right = n + 1
            window[0:4, 0:4] = NSML_in[I_up:I_bottom + 1, I_left:I_right + 1]
            window = omiga * window * window
            NSML_o[m, n] = torch.sum(window)

    NSML_out = NSML_o[1:H_n + 1, 1:W_n + 1]
    return NSML_out


def Get_NSML_Tensor(tensor):
    n_tensor = torch.zeros_like(tensor)
    B, C, W, H = tensor.shape
    for b in range(B):
        for c in range(C):
            n_tensor[b, c] = NSML((tensor[b, c] + 1) / 2.0)
    return n_tensor


class Clar_Loss(nn.Module):
    def __init__(self):
        super(Clar_Loss, self).__init__()

    def forward(self, TensorA, TensorB):
        LA = Get_NSML_Tensor(TensorA)
        LB = Get_NSML_Tensor(TensorB)
        diff = torch.nn.MSELoss()
        loss = diff(LA, LB)
        return loss


sampledataA = torch.zeros(1, 1, 256, 256)
sampledataB = torch.zeros(1, 1, 256, 256)
I1 = cv2.imread('./database/nyu/multifocus/background/train/766.png', 0) / 255.0 * 2 - 1
T1 = torch.from_numpy(I1)
sampledataA[0, 0] = T1
I2 = cv2.imread('./database/nyu/multifocus/truth/train/766.png', 0) / 255.0 * 2 - 1
T2 = torch.from_numpy(I2)
sampledataB[0, 0] = T2
# N1 = Get_NSML_Tensor(sampledataA)
# print(N1)
# NN = N1.numpy()
# cv2.imshow('N1', NN[0, 0])
# cv2.waitKey(0)
Net = Clar_Loss()
OUT = Net(sampledataA, sampledataB)
print(OUT)
