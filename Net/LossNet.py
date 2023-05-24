from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F


class L2_LOSS(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L2_LOSS, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        square = torch.square(diff)
        loss = torch.sum(square) / X.size(0)
        return loss


class L1_LOSS(nn.Module):
    def __init__(self):
        super(L1_LOSS, self).__init__()
        self.eps = 1e-6

    def forward(self, Ximage, Ytarget):
        diff = torch.add(Ximage, -Ytarget)
        # square = torch.square(diff)
        abs = torch.abs(diff)
        loss = torch.sum(abs) / Ximage.size(0)
        return loss

class L1_LOSS_Noreduce(nn.Module):
    def __init__(self):
        super(L1_LOSS_Noreduce, self).__init__()
        self.eps = 1e-6

    def forward(self, Ximage, Ytarget):
        diff = torch.add(Ximage, -Ytarget)
        # square = torch.square(diff)
        abs = torch.abs(diff)
        loss = torch.sum(torch.sum(torch.sum(abs, dim=-1), dim=-1), dim=-1)
        loss = loss / Ximage.shape[1] / Ximage.shape[2] / Ximage.shape[3]
        # loss = torch.sum(abs) / Ximage.size(0)
        return loss


class L1_Advanced_Sobel_Loss(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super().__init__()
        self.device = device
        self.conv_op_x = nn.Conv2d(3,3, 3, bias=False)
        self.conv_op_y = nn.Conv2d(3,3, 3, bias=False)

        sobel_kernel_x = np.array([[[2, 1, 0], [1, 0, -1], [0,-1, -2]],
                                   [[2, 1, 0], [1, 0, -1], [0,-1, -2]],
                                   [[2, 1, 0], [1, 0, -1], [0,-1, -2]]], dtype='float32')
        sobel_kernel_y = np.array([[[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
                                   [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
                                   [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]], dtype='float32')
        sobel_kernel_x = sobel_kernel_x.reshape((1, 3, 3, 3))
        sobel_kernel_y = sobel_kernel_y.reshape((1, 3, 3, 3))

        self.conv_op_x.weight.data = torch.from_numpy(sobel_kernel_x).to(device)
        self.conv_op_y.weight.data = torch.from_numpy(sobel_kernel_y).to(device)
        self.conv_op_x.weight.requires_grad = False
        self.conv_op_y.weight.requires_grad = False

    # def forward(self, edge_outputs, image_target):
    def forward(self, outputs, image_target):
        edge_Y_xoutputs = self.conv_op_x(outputs)
        edge_Y_youtputs = self.conv_op_y(outputs)
        edge_Youtputs = torch.abs(edge_Y_xoutputs) + torch.abs(edge_Y_youtputs)

        edge_Y_x = self.conv_op_x(image_target)
        edge_Y_y = self.conv_op_y(image_target)
        edge_Y = torch.abs(edge_Y_x) + torch.abs(edge_Y_y)

        diff = torch.add(edge_Youtputs, -edge_Y)
        error = torch.abs(diff)
        loss = torch.sum(error) / outputs.size(0)
        return loss


class L1_Advanced_Sobel_Loss_Noreduce(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super().__init__()
        self.device = device
        self.conv_op_x = nn.Conv2d(3,3, 3, bias=False)
        self.conv_op_y = nn.Conv2d(3,3, 3, bias=False)

        sobel_kernel_x = np.array([[[2, 1, 0], [1, 0, -1], [0,-1, -2]],
                                   [[2, 1, 0], [1, 0, -1], [0,-1, -2]],
                                   [[2, 1, 0], [1, 0, -1], [0,-1, -2]]], dtype='float32')
        sobel_kernel_y = np.array([[[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
                                   [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
                                   [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]], dtype='float32')
        sobel_kernel_x = sobel_kernel_x.reshape((1, 3, 3, 3))
        sobel_kernel_y = sobel_kernel_y.reshape((1, 3, 3, 3))

        self.conv_op_x.weight.data = torch.from_numpy(sobel_kernel_x).to(device)
        self.conv_op_y.weight.data = torch.from_numpy(sobel_kernel_y).to(device)
        self.conv_op_x.weight.requires_grad = False
        self.conv_op_y.weight.requires_grad = False

    # def forward(self, edge_outputs, image_target):
    def forward(self, outputs, image_target):
        edge_Y_xoutputs = self.conv_op_x(outputs)
        edge_Y_youtputs = self.conv_op_y(outputs)
        edge_Youtputs = torch.abs(edge_Y_xoutputs) + torch.abs(edge_Y_youtputs)

        edge_Y_x = self.conv_op_x(image_target)
        edge_Y_y = self.conv_op_y(image_target)
        edge_Y = torch.abs(edge_Y_x) + torch.abs(edge_Y_y)

        diff = torch.add(edge_Youtputs, -edge_Y)
        error = torch.abs(diff)
        loss = torch.sum(torch.sum(torch.sum(error, dim=-1), dim=-1), dim=-1)
        loss = loss / image_target.shape[1] / image_target.shape[2] / image_target.shape[3]
        # loss = torch.sum(error) / outputs.size(0)
        return loss

class L1_Sobel_Loss(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super(L1_Sobel_Loss, self).__init__()
        self.device = device
        self.conv_op_x = nn.Conv2d(3,3, 3, bias=False)
        self.conv_op_y = nn.Conv2d(3,3, 3, bias=False)

        sobel_kernel_x = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                   [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                   [[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype='float32')
        sobel_kernel_y = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                   [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                   [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], dtype='float32')
        sobel_kernel_x = sobel_kernel_x.reshape((1, 3, 3, 3))
        sobel_kernel_y = sobel_kernel_y.reshape((1, 3, 3, 3))

        self.conv_op_x.weight.data = torch.from_numpy(sobel_kernel_x).to(device)
        self.conv_op_y.weight.data = torch.from_numpy(sobel_kernel_y).to(device)
        self.conv_op_x.weight.requires_grad = False
        self.conv_op_y.weight.requires_grad = False

    # def forward(self, edge_outputs, image_target):
    def forward(self, outputs, image_target):
        edge_Y_xoutputs = self.conv_op_x(outputs)
        edge_Y_youtputs = self.conv_op_y(outputs)
        edge_Youtputs = torch.abs(edge_Y_xoutputs) + torch.abs(edge_Y_youtputs)

        edge_Y_x = self.conv_op_x(image_target)
        edge_Y_y = self.conv_op_y(image_target)
        edge_Y = torch.abs(edge_Y_x) + torch.abs(edge_Y_y)


        # diff = torch.add(edge_outputs, -edge_Y)
        diff = torch.add(edge_Youtputs, -edge_Y)
        error = torch.abs(diff)
        loss = torch.sum(error) #/ outputs.size(0) #output.size(0)은 batch size
        return loss


class edge_making(nn.Module):
    def __init__(self):
        super(edge_making, self).__init__()
        self.conv_op_x, self.conv_op_y = self.make_sobel_layer()

    def forward(self, output):
        output = (output*2)-1
        edge_X_x = self.conv_op_x(output)
        edge_X_y = self.conv_op_y(output)
        edge_X = torch.abs(edge_X_x) + torch.abs(edge_X_y)
        return edge_X

    def make_sobel_layer(self):
        conv_op_x = nn.Conv2d(3, 1, 3, bias=False)
        conv_op_y = nn.Conv2d(3, 1, 3, bias=False)

        sobel_kernel_x = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                   [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                                   [[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype='float32')
        sobel_kernel_y = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                   [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                   [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], dtype='float32')
        sobel_kernel_x = sobel_kernel_x.reshape((1, 3, 3, 3))
        sobel_kernel_y = sobel_kernel_y.reshape((1, 3, 3, 3))

        conv_op_x.weight.data = torch.from_numpy(sobel_kernel_x)
        conv_op_y.weight.data = torch.from_numpy(sobel_kernel_y)
        conv_op_x.weight.requires_grad = False
        conv_op_y.weight.requires_grad = False

        return conv_op_x, conv_op_y




class Sobelloss_L1(nn.Module):
    """edge_loss"""
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, image, target, cuda=True):
        x_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        y_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        # x_filter3 = np.zeros_like(x_filter)
        # y_filter3 = np.zeros_like(y_filter)
        x_filter3 = np.zeros((3,3,3))
        y_filter3 = np.zeros((3,3,3))
        x_filter3[:,:,0] = x_filter
        x_filter3[:,:,1] = x_filter
        x_filter3[:,:,2] = x_filter
        y_filter3[:,:,0] = y_filter
        y_filter3[:,:,1] = y_filter
        y_filter3[:,:,2] = y_filter

        convx = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        convy = nn.Conv2d(3, 3, kernel_size=3 , stride=1, padding=1, bias=False)
        weights_x = torch.from_numpy(x_filter3).float().unsqueeze(0).unsqueeze(0)
        weights_y = torch.from_numpy(y_filter3).float().unsqueeze(0).unsqueeze(0)

        if cuda:
            weights_x = weights_x.cuda()
            weights_y = weights_y.cuda()

        convx.weight = nn.Parameter(weights_x)
        convy.weight = nn.Parameter(weights_y)

        convx.weight.requires_grad = False
        convy.weight.requires_grad = False

        g1_x = convx(image)
        g2_x = convx(target)
        g1_y = convy(image)
        g2_y = convy(target)

        g_1 = torch.sqrt(torch.pow(g1_x, 2) + torch.pow(g1_y, 2))
        g_2 = torch.sqrt(torch.pow(g2_x, 2) + torch.pow(g2_y, 2))

        loss = torch.sqrt((g_1 - g_2).pow(2))
        loss = torch.sum(loss) / image.size(0)

        # return torch.mean((g_1 - g_2).pow(2)) # L2, MSE loss
        # return torch.sqrt((g_1 - g_2).pow(2)) # L1sobel loss
        return loss # L1sobel loss


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6
    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error) / X.size(0)
        return loss




def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD

class multi_VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, lam=1, lam_p=1, reduce=True):
        super(multi_VGGPerceptualLoss, self).__init__()
        self.loss_fn = VGGPerceptualLoss(reduce=reduce)
        self.lam = lam
        self.lam_p = lam_p
        self.reduce = reduce

    def forward(self, out1, out2, out3, gt1, feature_layers=[2]):
        gt2 = F.interpolate(gt1, scale_factor=0.5, mode='bilinear', align_corners=False)
        gt3 = F.interpolate(gt1, scale_factor=0.25, mode='bilinear', align_corners=False)

        if self.reduce:
            loss1 = self.lam_p * self.loss_fn(out1, gt1, feature_layers=feature_layers) + \
                    self.lam * F.l1_loss(out1, gt1)
            loss2 = self.lam_p * self.loss_fn(out2, gt2, feature_layers=feature_layers) + \
                    self.lam * F.l1_loss(out2, gt2)
            loss3 = self.lam_p * self.loss_fn(out3, gt3, feature_layers=feature_layers) + \
                    self.lam * F.l1_loss(out3, gt3)
        else:
            loss1 = self.lam_p * self.loss_fn(out1, gt1, feature_layers=feature_layers) + \
                    self.lam * \
                    torch.sum(torch.sum(torch.sum(F.l1_loss(out1, gt1, reduction='none'), dim=-1), dim=-1), dim=-1) \
                    / out1.shape[1] / out1.shape[2] / out1.shape[3]
            loss2 = self.lam_p * self.loss_fn(out2, gt2, feature_layers=feature_layers) + \
                    self.lam * \
                    torch.sum(torch.sum(torch.sum(F.l1_loss(out2, gt2, reduction='none'), dim=-1), dim=-1), dim=-1) \
                    / out1.shape[1] / out1.shape[2] / out1.shape[3]
            loss3 = self.lam_p * self.loss_fn(out3, gt3, feature_layers=feature_layers) + \
                    self.lam * \
                    torch.sum(torch.sum(torch.sum(F.l1_loss(out3, gt3, reduction='none'), dim=-1), dim=-1), dim=-1) \
                    / out1.shape[1] / out1.shape[2] / out1.shape[3]
        return loss1 + loss2 + loss3


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True, reduce=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize
        self.reduce = reduce

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        if self.reduce:
            loss = 0.0
            x = input
            y = target
            for i, block in enumerate(self.blocks):
                x = block(x)
                y = block(y)
                if i in feature_layers:
                    loss += torch.nn.functional.l1_loss(x, y)
                if i in style_layers:
                    act_x = x.reshape(x.shape[0], x.shape[1], -1)
                    act_y = y.reshape(y.shape[0], y.shape[1], -1)
                    gram_x = act_x @ act_x.permute(0, 2, 1)
                    gram_y = act_y @ act_y.permute(0, 2, 1)
                    loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        else:
            loss = torch.zeros(len(input)).cuda()
            x = input
            y = target
            for i, block in enumerate(self.blocks):
                x = block(x)
                y = block(y)
                if i in feature_layers:
                    tmp = torch.nn.functional.l1_loss(x, y, reduction='none')
                    tmp =  torch.sum(torch.sum(torch.sum(tmp, dim=-1), dim=-1), dim=-1)
                    tmp = tmp / x.shape[1] / x.shape[2] / x.shape[3]
                    loss += tmp
                if i in style_layers:
                    act_x = x.reshape(x.shape[0], x.shape[1], -1)
                    act_y = y.reshape(y.shape[0], y.shape[1], -1)
                    gram_x = act_x @ act_x.permute(0, 2, 1)
                    gram_y = act_y @ act_y.permute(0, 2, 1)
                    tmp = torch.nn.functional.l1_loss(gram_x, gram_y, reduction='none')
                    tmp = torch.sum(torch.sum(torch.sum(tmp, dim=-1), dim=-1), dim=-1)
                    tmp = tmp / gram_x.shape[1] / gram_x.shape[2] / gram_x.shape[3]
                    loss += tmp
        return loss