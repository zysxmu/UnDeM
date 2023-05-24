import os
import random

# import h5py
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from pathlib import Path
import pdb
import random
import pickle
from skimage import color
from skimage import io

def default_loader(path):
    img = Image.open(path).convert('RGB')
    w, h = img.size

    return img

def default_loader_crop(path):
    img = Image.open(path).convert('RGB')
    region = img.crop( (156,156,660,660) )
    return region

class Moire_dataset(Dataset):
    def __init__(self, root, loader = default_loader):
        moire_data_root = os.path.join(root, 'source')
        clear_data_root = os.path.join(root, 'target')

        image_names = os.listdir(clear_data_root)
        image_names = ["_".join(i.split("_")[:-1]) for i in image_names]

        self.moire_images = [os.path.join(moire_data_root, x + '_source.png') for x in image_names]
        self.clear_images = [os.path.join(clear_data_root, x + '_target.png') for x in image_names]
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.loader = loader
        self.labels = image_names

    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]
        clear_img_path = self.clear_images[index]
        moire = self.loader(moire_img_path)
        clear = self.loader(clear_img_path)
        moire = self.transforms(moire)
        clear = self.transforms(clear)
        label = self.labels[index]
        return moire, clear, label

    def __len__(self):
        return len(self.moire_images)


class FHDMI_dataset(Dataset):
    def __init__(self, root, crop = True, loader = default_loader, resize=None, patch_size=384):
        self.crop = crop
        moire_data_root = os.path.join(root, 'moire')
        clear_data_root = os.path.join(root, 'clear')
        self.patch_size = patch_size
        image_names1 = os.listdir(moire_data_root)
        image_names1.sort()
        self.moire_images = [os.path.join(moire_data_root, x ) for x in image_names1]

        image_names2 = os.listdir(clear_data_root)
        image_names2.sort()
        self.clear_images = [os.path.join(clear_data_root, x ) for x in image_names2]

        image_names1 = [".".join(i.split(".")[:-1]) for i in image_names1]

        self.transforms = transforms.Compose([
            # transforms.Resize((400, 200)),
            transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transforms2 = transforms.Compose([
            transforms.Resize((patch_size//2, patch_size//2)),
        ])
        self.transforms3 = transforms.Compose([
            transforms.Resize((patch_size//4, patch_size//4)),
        ])
        # self.transforms3 = transforms.CenterCrop(128,128)


        self.loader = loader
        self.labels = image_names1

    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]

        new_index = random.randint(0, len(self.moire_images)-1)
        while new_index == index:
            new_index = random.randint(0, len(self.moire_images) - 1)
        clear_img_path = self.clear_images[new_index]
        moire = self.loader(moire_img_path)
        clear = self.loader(clear_img_path)
        moire = self.transforms(moire)
        clear = self.transforms(clear)


        if self.crop == True:
            i, j, h, w = transforms.RandomCrop.get_params(moire, output_size=(self.patch_size, self.patch_size))

            i2, j2, h2, w2 = transforms.RandomCrop.get_params(moire, output_size=(self.patch_size, self.patch_size))

            moire2 = TF.crop(moire, i2, j2, h2, w2)
            clear2 = TF.crop(clear, i2, j2, h2, w2)

            moire = TF.crop(moire, i, j, h, w)
            clear = TF.crop(clear, i, j, h, w)


        # clear2 = self.transforms2(clear)
        # clear3 = self.transforms3(clear2)
        # clear_list = [clear3, clear2, clear]
        label = self.labels[index]
        return moire, clear, moire2, clear2

    def __len__(self):
        return len(self.moire_images)

# image testteststest
class FHDMI_dataset_test(Dataset):
    def __init__(self, root, crop = False, loader = default_loader,resize=None):
        self.crop = crop
        moire_data_root = os.path.join(root, 'moire')
        clear_data_root = os.path.join(root, 'clear')

        image_names1 = os.listdir(moire_data_root)
        image_names1.sort()
        self.moire_images = [os.path.join(moire_data_root, x ) for x in image_names1]

        image_names2 = os.listdir(clear_data_root)
        image_names2.sort()
        self.clear_images = [os.path.join(clear_data_root, x ) for x in image_names2]

        image_names1 = [".".join(i.split(".")[:-1]) for i in image_names1]

        self.transforms = transforms.Compose([
            transforms.Resize((1024, 1920)),
            transforms.ToTensor(),

        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # self.transforms2 = transforms.Compose([
        #     transforms.Resize((512, 512)),
        # ])
        # self.transforms3 = transforms.Compose([
        #     transforms.Resize((256, 256)),
        # ])

        self.loader = loader
        self.labels = image_names1

    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]
        clear_img_path = self.clear_images[index]
        moire = self.loader(moire_img_path)
        clear = self.loader(clear_img_path)
        moire = self.transforms(moire)
        clear = self.transforms(clear)


        # clear2 = self.transforms2(clear)
        # clear3 = self.transforms3(clear2)

        # clear2 = self.transforms2(clear)
        # clear3 = self.transforms3(clear2)

        # clear_list = clear
        label = self.labels[index]
        return moire, clear, label

    def __len__(self):
        return len(self.moire_images)


class UHDM_dataset(Dataset):
    def __init__(self, root, crop = True, loader = default_loader, resize=None, patch_size=384):
        self.crop = crop
        moire_data_root = os.path.join(root, 'moire')
        clear_data_root = os.path.join(root, 'clear')
        self.patch_size = patch_size
        image_names1 = os.listdir(moire_data_root)
        image_names1.sort()
        self.moire_images = [os.path.join(moire_data_root, x ) for x in image_names1]

        image_names2 = os.listdir(clear_data_root)
        image_names2.sort()
        self.clear_images = [os.path.join(clear_data_root, x ) for x in image_names2]

        image_names1 = [".".join(i.split(".")[:-1]) for i in image_names1]

        self.transforms = transforms.Compose([
            # transforms.Resize((400, 200)),
            transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transforms2 = transforms.Compose([
            transforms.Resize((patch_size//2, patch_size//2)),
        ])
        self.transforms3 = transforms.Compose([
            transforms.Resize((patch_size//4, patch_size//4)),
        ])
        # self.transforms3 = transforms.CenterCrop(128,128)


        self.loader = loader
        self.labels = image_names1

    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]

        new_index = random.randint(0, len(self.moire_images)-1)
        while new_index == index:
            new_index = random.randint(0, len(self.moire_images) - 1)
        clear_img_path = self.clear_images[new_index]
        moire = self.loader(moire_img_path)
        clear = self.loader(clear_img_path)
        moire = self.transforms(moire)
        clear = self.transforms(clear)


        if self.crop == True:
            i, j, h, w = transforms.RandomCrop.get_params(moire, output_size=(self.patch_size, self.patch_size))

            i2, j2, h2, w2 = transforms.RandomCrop.get_params(moire, output_size=(self.patch_size, self.patch_size))

            moire2 = TF.crop(moire, i2, j2, h2, w2)
            clear2 = TF.crop(clear, i2, j2, h2, w2)

            moire = TF.crop(moire, i, j, h, w)
            clear = TF.crop(clear, i, j, h, w)


        # clear2 = self.transforms2(clear)
        # clear3 = self.transforms3(clear2)
        # clear_list = [clear3, clear2, clear]
        label = self.labels[index]
        return moire, clear, moire2, clear2

    def __len__(self):
        return len(self.moire_images)

# image testteststest
class UHDM_dataset_test(Dataset):
    def __init__(self, root, crop = False, loader = default_loader,resize=None):
        self.crop = crop
        moire_data_root = os.path.join(root, 'moire')
        clear_data_root = os.path.join(root, 'clear')

        image_names1 = os.listdir(moire_data_root)
        image_names1.sort()
        self.moire_images = [os.path.join(moire_data_root, x ) for x in image_names1]

        image_names2 = os.listdir(clear_data_root)
        image_names2.sort()
        self.clear_images = [os.path.join(clear_data_root, x ) for x in image_names2]

        image_names1 = [".".join(i.split(".")[:-1]) for i in image_names1]

        self.transforms = transforms.Compose([
            transforms.Resize((1024, 1920)),
            transforms.ToTensor(),

        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # self.transforms2 = transforms.Compose([
        #     transforms.Resize((512, 512)),
        # ])
        # self.transforms3 = transforms.Compose([
        #     transforms.Resize((256, 256)),
        # ])

        self.loader = loader
        self.labels = image_names1

    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]
        clear_img_path = self.clear_images[index]
        moire = self.loader(moire_img_path)
        clear = self.loader(clear_img_path)
        moire = self.transforms(moire)
        clear = self.transforms(clear)


        # clear2 = self.transforms2(clear)
        # clear3 = self.transforms3(clear2)

        # clear2 = self.transforms2(clear)
        # clear3 = self.transforms3(clear2)

        clear_list = [torch.tensor(0), torch.tensor(0), clear]

        # clear_list = clear
        label = self.labels[index]
        return moire, clear_list, label

    def __len__(self):
        return len(self.moire_images)

