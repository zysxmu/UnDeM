import os
import time
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from Util.util_collections import tensor2im, save_single_image, PSNR, Time2Str, setup_logging, \
    CosineAnnealingWarmRestarts

from dataset.dataset_weeksup import FHDMI_dataset, UHDM_dataset

from skimage.metrics import peak_signal_noise_ratio
import math
import logging
import random
from model import define_models
import cv2
import torchvision
import tqdm
import argparse


def load_model(netG_mo_class, encoder_mo_class, path):
    path_1 = 'path_for_netG_mo' # set your path here
    print('load: ' + path_1)
    netG_mo_class.load_state_dict(torch.load(path_1))

    path_1 = 'path_for_encoder_mo' # set your path here
    print('load: ' + path_1)
    encoder_mo_class.load_state_dict(torch.load(path_1))

    return netG_mo_class, encoder_mo_class


def get_dataloader(args, traindata_path):
    if args.dataset == 'fhdmi':
        train_dataset = FHDMI_dataset
    elif args.dataset == 'uhdm':
        train_dataset = UHDM_dataset
    else:
        raise ValueError('no this dataset choise')

    Moiredata_train = train_dataset(traindata_path, crop=False)
    train_dataloader = DataLoader(Moiredata_train,
                                  batch_size=args.batchsize,
                                  shuffle=True,
                                  num_workers=args.num_worker,
                                  drop_last=True)
    return train_dataloader

def get_pretrain_generation_model(args):
    # Networks
    _, netG_mo_class1, _, _, encoder_mo_class1, _ = define_models()
    _, netG_mo_class2, _, _, encoder_mo_class2, _ = define_models()
    _, netG_mo_class3, _, _, encoder_mo_class3, _ = define_models()
    _, netG_mo_class4, _, _, encoder_mo_class4, _ = define_models()

    if args.dataset == 'fhdmi':
        if args.patch_size == 384:
            netG_mo_class1, encoder_mo_class1 = load_model(netG_mo_class1, encoder_mo_class1,
                                                           'path_for_fhdmi_moire_generation_model_of_class1_patch384')
            netG_mo_class2, encoder_mo_class2 = load_model(netG_mo_class2, encoder_mo_class2,
                                                           'path_for_fhdmi_moire_generation_model_of_class2_patch384')
            netG_mo_class3, encoder_mo_class3 = load_model(netG_mo_class3, encoder_mo_class3,
                                                           'path_for_fhdmi_moire_generation_model_of_class3_patch384')
            netG_mo_class4, encoder_mo_class4 = load_model(netG_mo_class4, encoder_mo_class4,
                                                           'path_for_fhdmi_moire_generation_model_of_class4_patch384')
        elif args.patch_size == 192:
            netG_mo_class1, encoder_mo_class1 = load_model(netG_mo_class1, encoder_mo_class1,
                                                           'path_for_fhdmi_moire_generation_model_of_class1_patch192')
            netG_mo_class2, encoder_mo_class2 = load_model(netG_mo_class2, encoder_mo_class2,
                                                           'path_for_fhdmi_moire_generation_model_of_class2_patch192')
            netG_mo_class3, encoder_mo_class3 = load_model(netG_mo_class3, encoder_mo_class3,
                                                           'path_for_fhdmi_moire_generation_model_of_class3_patch192')
            netG_mo_class4, encoder_mo_class4 = load_model(netG_mo_class4, encoder_mo_class4,
                                                           'path_for_fhdmi_moire_generation_model_of_class4_patch192')
    elif args.dataset == 'uhdm':
        if args.patch_size == 192:
            netG_mo_class1, encoder_mo_class1 = load_model(netG_mo_class1, encoder_mo_class1,
                                                           'path_for_uhdm_moire_generation_model_of_class1_patch192')
            netG_mo_class2, encoder_mo_class2 = load_model(netG_mo_class2, encoder_mo_class2,
                                                           'path_for_uhdm_moire_generation_model_of_class2_patch192')
            netG_mo_class3, encoder_mo_class3 = load_model(netG_mo_class3, encoder_mo_class3,
                                                           'path_for_uhdm_moire_generation_model_of_class3_patch192')
            netG_mo_class4, encoder_mo_class4 = load_model(netG_mo_class4, encoder_mo_class4,
                                                           'path_for_uhdm_moire_generation_model_of_class4_patch192')
        elif args.patch_size == 384:
            netG_mo_class1, encoder_mo_class1 = load_model(netG_mo_class1, encoder_mo_class1,
                                                           'path_for_uhdm_moire_generation_model_of_class1_patch384')
            netG_mo_class2, encoder_mo_class2 = load_model(netG_mo_class2, encoder_mo_class2,
                                                           'path_for_uhdm_moire_generation_model_of_class2_patch384')
            netG_mo_class3, encoder_mo_class3 = load_model(netG_mo_class3, encoder_mo_class3,
                                                           'path_for_uhdm_moire_generation_model_of_class3_patch384')
            netG_mo_class4, encoder_mo_class4 = load_model(netG_mo_class4, encoder_mo_class4,
                                                           'path_for_uhdm_moire_generation_model_of_class4_patch384')
        elif args.patch_size == 768:
            netG_mo_class1, encoder_mo_class1 = load_model(netG_mo_class1, encoder_mo_class1,
                                                           'path_for_uhdm_moire_generation_model_of_class1_patch768')
            netG_mo_class2, encoder_mo_class2 = load_model(netG_mo_class2, encoder_mo_class2,
                                                           'path_for_uhdm_moire_generation_model_of_class2_patch768')
            netG_mo_class3, encoder_mo_class3 = load_model(netG_mo_class3, encoder_mo_class3,
                                                           'path_for_uhdm_moire_generation_model_of_class3_patch768')
            netG_mo_class4, encoder_mo_class4 = load_model(netG_mo_class4, encoder_mo_class4,
                                                           'path_for_uhdm_moire_generation_model_of_class4_patch768')
    return  netG_mo_class1, netG_mo_class2, netG_mo_class3, netG_mo_class4, \
            encoder_mo_class1, encoder_mo_class2, encoder_mo_class3, encoder_mo_class4


def laplacian(image):

    res = []

    a = tensor2im(image)
    for i in range(len(a)):

        img = a[i]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplac = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
        mask_img = cv2.convertScaleAbs(laplac)
        res.append(torch.mean(torch.Tensor(mask_img)))

    return torch.tensor(res)


def laplacian_diff(image, image_1):

    res = []

    a = tensor2im(image)
    a_1 = tensor2im(image_1)
    for i in range(len(a)):

        img = a[i]
        img_1 = a_1[i]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplac = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
        mask_img = cv2.convertScaleAbs(laplac)

        gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        laplac_1 = cv2.Laplacian(gray_1, cv2.CV_16S, ksize=3)
        mask_img_1 = cv2.convertScaleAbs(laplac_1)

        diff = torch.abs(torch.Tensor(mask_img) - torch.Tensor(mask_img_1))
        res.append(torch.mean(diff))

    return torch.tensor(res)


def get_sort(args):
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Networks
    _, netG_mo_class1, _, _, encoder_mo_class1, _ = define_models()
    _, netG_mo_class2, _, _, encoder_mo_class2, _ = define_models()
    _, netG_mo_class3, _, _, encoder_mo_class3, _ = define_models()
    _, netG_mo_class4, _, _, encoder_mo_class4, _ = define_models()

    netG_mo_class1, netG_mo_class2, netG_mo_class3, netG_mo_class4, \
    encoder_mo_class1, encoder_mo_class2, encoder_mo_class3, encoder_mo_class4 = get_pretrain_generation_model(args)

    netG_mo_class1.eval()
    netG_mo_class2.eval()
    netG_mo_class3.eval()
    netG_mo_class4.eval()
    encoder_mo_class1.eval()
    encoder_mo_class2.eval()
    encoder_mo_class3.eval()
    encoder_mo_class4.eval()

    train_dataloader_class1 = get_dataloader(args, 'traindata_path_of_pacthes_class1')
    train_dataloader_class2 = get_dataloader(args, 'traindata_path_of_pacthes_class2')
    train_dataloader_class3 = get_dataloader(args, 'traindata_path_of_pacthes_class3')
    train_dataloader_class4 = get_dataloader(args, 'traindata_path_of_pacthes_class4')

    encoder_mo_class1 = nn.DataParallel(encoder_mo_class1)
    encoder_mo_class2 = nn.DataParallel(encoder_mo_class2)
    encoder_mo_class3 = nn.DataParallel(encoder_mo_class3)
    encoder_mo_class4 = nn.DataParallel(encoder_mo_class4)
    netG_mo_class1 = nn.DataParallel(netG_mo_class1)
    netG_mo_class2 = nn.DataParallel(netG_mo_class2)
    netG_mo_class3 = nn.DataParallel(netG_mo_class3)
    netG_mo_class4 = nn.DataParallel(netG_mo_class4)

    train_loader_1 = iter(train_dataloader_class1)
    train_loader_2 = iter(train_dataloader_class2)
    train_loader_3 = iter(train_dataloader_class3)
    train_loader_4 = iter(train_dataloader_class4)
    print(len(train_loader_1), len(train_loader_2), len(train_loader_3), len(train_loader_4))
    c = args.class_index
    llll = []
    for j in tqdm.tqdm(range(400)):
        # print(c)
        with torch.no_grad():
            if c == 0:
                try:
                    moires, clear, _, _, clear2, clear3 = next(train_loader_1)
                except:
                    print('re-iterator of train_loader_1')
                    train_loader_1 = iter(train_dataloader_class1)
                    moires, clear, _, _, clear2, clear3 = next(train_loader_1)
                moires = moires.cuda()
                clear = clear.cuda()
                clear2, clear3 = clear2.cuda(), clear3.cuda()
                real_mo_feat = encoder_mo_class1(moires)
                fake_mo = netG_mo_class1(real_mo_feat, clear)
            elif c == 1:
                try:
                    moires, clear, _, _, clear2, clear3 = next(train_loader_2)
                except:
                    print('re-iterator of train_loader_2')
                    train_loader_2 = iter(train_dataloader_class2)
                    moires, clear, _, _, clear2, clear3 = next(train_loader_2)
                moires = moires.cuda()
                clear = clear.cuda()
                clear2, clear3 = clear2.cuda(), clear3.cuda()
                real_mo_feat = encoder_mo_class2(moires)
                fake_mo = netG_mo_class2(real_mo_feat, clear)
            elif c == 2:
                try:
                    moires, clear, _, _, clear2, clear3 = next(train_loader_3)
                except:
                    print('re-iterator of train_loader_3')
                    train_loader_3 = iter(train_dataloader_class3)
                    moires, clear, _, _, clear2, clear3 = next(train_loader_3)
                moires = moires.cuda()
                clear = clear.cuda()
                clear2, clear3 = clear2.cuda(), clear3.cuda()
                real_mo_feat = encoder_mo_class3(moires)
                fake_mo = netG_mo_class3(real_mo_feat, clear)
            elif c == 3:
                try:
                    moires, clear, _, _, clear2, clear3 = next(train_loader_4)
                except:
                    print('re-iterator of train_loader_4')
                    train_loader_4 = iter(train_dataloader_class4)
                    moires, clear, _, _, clear2, clear3 = next(train_loader_4)
                moires = moires.cuda()
                clear = clear.cuda()
                clear2, clear3 = clear2.cuda(), clear3.cuda()
                real_mo_feat = encoder_mo_class4(moires)
                fake_mo = netG_mo_class4(real_mo_feat, clear)

            clear1 = clear

        clear1, clear2, clear3, moires, fake_mo = \
            clear1.detach(), clear2.detach(), clear3.detach(), moires.detach(), fake_mo.detach()

        a = laplacian_diff(clear1, fake_mo)
        llll.append(a)

    print(c)
    llll = torch.concat(llll)
    print(len(llll))
    topk, _ = torch.topk(llll, int(len(llll) * 0.005))
    print(0.5, "%", topk[-1].cpu().numpy())
    topk, _ = torch.topk(llll, int(len(llll) * 0.01))
    print(1, "%", topk[-1].cpu().numpy())
    topk, _ = torch.topk(llll, int(len(llll) * 0.05))
    print(5, "%", topk[-1].cpu().numpy())
    topk, _ = torch.topk(llll, int(len(llll) * 0.1))
    print(10, "%", topk[-1].cpu().numpy())
    topk, _ = torch.topk(llll, int(len(llll) * 0.2))
    print(20, "%", topk[-1].cpu().numpy())
    topk, _ = torch.topk(llll, int(len(llll) * 0.3))
    print(30, "%", topk[-1].cpu().numpy())
    topk, _ = torch.topk(llll, int(len(llll) * 0.4))
    print(40, "%", topk[-1].cpu().numpy())
    topk, _ = torch.topk(llll, int(len(llll) * 0.5))
    print(50, "%", topk[-1].cpu().numpy())
    import sys
    sys.exit()

    return "Training Finished!"


parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=16, help='size of the batches')
parser.add_argument('--num_worker', type=int, default=8,
                    help='number of workers')
parser.add_argument('--dataset',default='uhdm',help='set dataset')
parser.add_argument('--patch_size',default=128,type=int,help='set patch size') # set the patch_size!
parser.add_argument('--class_index',default=1,type=int,help='which patch class to be process') # set the patch_size!


args = parser.parse_args()

if __name__ == "__main__":
    get_sort(args)