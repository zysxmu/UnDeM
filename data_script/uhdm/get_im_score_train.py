"""A multi-thread tool to crop large images to sub-images for faster IO."""
"""https://github.com/XPixelGroup/ClassSR/blob/main/codes/data_scripts/extract_subimages_train.py"""
import os
import os.path as osp
import sys

from multiprocessing import Pool
import numpy as np
import cv2
from PIL import Image
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from utils import ProgressBar  # noqa: E402
import utils as data_util  # noqa: E402
import time
import torch
import pdb
import shutil

save_list=["/data/uhdm_class/train/class1/moire",
           "/data/uhdm_class/train/class2/moire",
           "/data/uhdm_class/train/class3/moire",
           "/data/uhdm_class/train/class4/moire",# fix to your path
           ]
clear_folder="/data/uhdm_patches/train/clear/"# fix to your path
moire_folder="/data/uhdm_patches/train/moire/"# fix to your path
score_list = {}
for i in save_list:
    if os.path.exists(i):
        pass
    else:
        os.makedirs(i)

def main():

    opt = {}
    opt['n_thread'] = 20
    # cut training data
    ########################################################################
    # check that all the clear and moire images have correct scale ratio
    img_moire_list = data_util._get_paths_from_images(moire_folder)
    img_clear_list = data_util._get_paths_from_images(clear_folder)
    
    print('processing...')
    opt['clear_folder'] = clear_folder
    opt['moire_folder'] = moire_folder
    extract_signle(opt)
       

def extract_signle(opt):
    input_folder = opt['moire_folder']

    img_list = data_util._get_paths_from_images(input_folder)
    #torch.multiprocessing.set_sharing_strategy('file_system')
        
    pbar = ProgressBar(len(img_list))
    #pool = Pool(20)

    for path in img_list:
        result = worker(path)
        score_list[result[0]] = (result[1], result[2], result[3], result[4])
        pbar.update(result[5])

    sort_result = sorted(score_list.items(), key=lambda x: (x[1][2]),
                         reverse=False)

    class_1 = sort_result[:27000//4]  # f small, c small
    sort_result = sort_result[27000//4:]

    sort_result = sorted(sort_result, key=lambda x: (x[1][3]), reverse=False)
    class_2 = sort_result[:27000//4]
    class_3 = sort_result[27000//4*1:27000//4*2]
    class_4 = sort_result[27000//4*2:]


    for i in range(0, len(class_1)):
        shutil.copy(osp.join(moire_folder, class_1[i][0]), osp.join(save_list[0], class_1[i][0]))
        shutil.copy(osp.join(clear_folder, class_1[i][0].replace('moire', 'gt')),
                    osp.join(save_list[8], class_1[i][0].replace('moire', 'gt')))
    for i in range(0, len(class_2)):
        shutil.copy(osp.join(moire_folder, class_2[i][0]), osp.join(save_list[1], class_2[i][0]))
        shutil.copy(osp.join(clear_folder, class_2[i][0].replace('moire', 'gt')),
                    osp.join(save_list[9], class_2[i][0].replace('moire', 'gt')))
    for i in range(0, len(class_3)):
        shutil.copy(osp.join(moire_folder, class_3[i][0]), osp.join(save_list[2], class_3[i][0]))
        shutil.copy(osp.join(clear_folder, class_3[i][0].replace('moire', 'gt')),
                    osp.join(save_list[10], class_3[i][0].replace('moire', 'gt')))
    for i in range(0, len(class_4)):
        shutil.copy(osp.join(moire_folder, class_4[i][0]), osp.join(save_list[3], class_4[i][0]))
        shutil.copy(osp.join(clear_folder, class_4[i][0].replace('moire', 'gt')),
                    osp.join(save_list[11], class_4[i][0].replace('moire', 'gt')))


def get_color(img):
    img = cv2.resize(img, (256, 256))
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    rg = torch.Tensor(R - G)
    yb = torch.Tensor(0.5 * (R + G) - B)

    dev_rg = torch.std(rg)
    dev_yb = torch.std(yb)
    mean_rg = torch.mean(rg)
    mean_yb = torch.mean(yb)
    dev = torch.sqrt(dev_rg * dev_rg + dev_yb * dev_yb)
    mean = torch.sqrt(mean_rg * mean_rg + mean_yb * mean_yb)
    M = dev + 0.3 * mean
    return M


def laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplac = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    mask_img = cv2.convertScaleAbs(laplac)
    return torch.mean(torch.Tensor(mask_img))



def im_score(image):
    color = get_color(image)
    edge = laplacian(image)
    # im_score = color * edge
    return color, edge

def worker(path):
    img_name = osp.basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    color, edge = im_score(img)
    # print("Img {} Im score {}".format(path, score))
    return [img_name, color, edge, color*edge, edge/color, 'Processing {:s} ...'.format(img_name)]



if __name__ == '__main__':
    main()