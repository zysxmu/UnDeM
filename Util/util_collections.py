import os
import numpy as np
import torch
import torch.nn as nn
import colour
from PIL import Image
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from torch.autograd import Variable
import math
import time
import logging
from torch.optim.lr_scheduler import _LRScheduler

def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
def setup_logging(log_file='log.txt',filemode='w'):
    """Setup logging configuration
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode=filemode)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def crop_cpu(img, crop_sz, step):
    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    index = 0
    num_h = 0
    lr_list=[]
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            lr_list.append(crop_img)
    h=x + crop_sz
    w=y + crop_sz
    return lr_list, num_h, num_w, h, w

def combine(sr_list, num_h, num_w, h, w, patch_size, step):
    index=0
    sr_img = np.zeros((h*self.scale, w*self.scale, 3), 'float32')
    for i in range(num_h):
        for j in range(num_w):
            sr_img[i*step*self.scale:i*step*self.scale+patch_size*self.scale,j*step*self.scale:j*step*self.scale+patch_size*self.scale,:]+=sr_list[index]
            index+=1
    sr_img=sr_img.astype('float32')

    for j in range(1,num_w):
        sr_img[:,j*step*self.scale:j*step*self.scale+(patch_size-step)*self.scale,:]/=2

    for i in range(1,num_h):
        sr_img[i*step*self.scale:i*step*self.scale+(patch_size-step)*self.scale,:,:]/=2
    return sr_img

def combine_addmask(self, sr_list, num_h, num_w, h, w, patch_size, step, _type):
    index = 0
    sr_img = np.zeros((h * self.scale, w * self.scale, 3), 'float32')

    for i in range(num_h):
        for j in range(num_w):
            sr_img[i * step * self.scale:i * step * self.scale + patch_size * self.scale,
            j * step * self.scale:j * step * self.scale + patch_size * self.scale, :] += sr_list[index]
            index += 1
    sr_img = sr_img.astype('float32')

    for j in range(1, num_w):
        sr_img[:, j * step * self.scale:j * step * self.scale + (patch_size - step) * self.scale, :] /= 2

    for i in range(1, num_h):
        sr_img[i * step * self.scale:i * step * self.scale + (patch_size - step) * self.scale, :, :] /= 2

    index2 = 0
    for i in range(num_h):
        for j in range(num_w):
            # add_mask
            alpha = 1
            beta = 0.2
            gamma = 0
            bbox1 = [j * step * self.scale + 8, i * step * self.scale + 8,
                        j * step * self.scale + patch_size * self.scale - 9,
                        i * step * self.scale + patch_size * self.scale - 9]  # xl,yl,xr,yr
            zeros1 = np.zeros((sr_img.shape), 'float32')

            if torch.max(_type, 1)[1].data.squeeze()[index2] == 0:
                mask2 = cv2.rectangle(zeros1, (bbox1[0]+1, bbox1[1]+1), (bbox1[2]-1, bbox1[3]-1), color=(0, 255, 0), thickness=-1)# simple green
            elif torch.max(_type, 1)[1].data.squeeze()[index2] == 1:
                mask2 = cv2.rectangle(zeros1, (bbox1[0]+1, bbox1[1]+1), (bbox1[2]-1, bbox1[3]-1), color=(0, 255, 255), thickness=-1)# medium yellow
            elif torch.max(_type, 1)[1].data.squeeze()[index2] == 2:
                mask2 = cv2.rectangle(zeros1, (bbox1[0]+1, bbox1[1]+1), (bbox1[2]-1, bbox1[3]-1), color=(0, 0, 255), thickness=-1)# hard red

            sr_img = cv2.addWeighted(sr_img, alpha, mask2, beta, gamma)
            index2+=1
    return sr_img

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in [50, 100, 150,200]:
        state['lr'] *= 0.3
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


def Time2Str():
    sec = time.time()
    tm = time.localtime(sec)
    time_str = '21'+'{:02d}'.format(tm.tm_mon)+'{:02d}'.format(tm.tm_mday) +'_'+'{:02d}'.format(tm.tm_hour+9)+':{:02d}'.format(tm.tm_min)
    return time_str


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in [50, 100, 150,200]:
        state['lr'] *= 0.3
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

def img_pad(x, w_pad, h_pad, w_odd_pad, h_odd_pad):
    '''
    Here the padding values are determined by the average r,g,b values across the training set
    in FHDMi dataset. For the evaluation on the UHDM, you can also try the commented lines where
    the mean values are calculated from UHDM training set, yielding similar performance.
    '''
    import torch.nn.functional as F
    x1 = F.pad(x[:, 0:1, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value=0.3827)
    x2 = F.pad(x[:, 1:2, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value=0.4141)
    x3 = F.pad(x[:, 2:3, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value=0.3912)
    # x1 = F.pad(x[:, 0:1, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value = 0.5165)
    # x2 = F.pad(x[:, 1:2, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value = 0.4952)
    # x3 = F.pad(x[:, 2:3, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value = 0.4695)
    y = torch.cat([x1, x2, x3], dim=1)

    return y

def crop(img, args):
    img = tensor2im(torch.squeeze(img))
    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))

    crop_sz_h = 1088
    crop_sz_w = 1280
    h_space = [0, 1088]
    w_space = [0, 1280, 2560]
    index = 0
    moire_list = []
    score_list = []
    for x in h_space:
        for y in w_space:
            if n_channels == 2:
                crop_img = img[x:x + crop_sz_h, y:y + crop_sz_w]
            else:
                crop_img = img[x:x + crop_sz_h, y:y + crop_sz_w, :]
            moire_list.append(crop_img)

    return moire_list, h_space, w_space, crop_sz_h, crop_sz_w


def combine(output_list, h_space, w_space, crop_sz_h, crop_sz_w, args):
    clear_img = torch.zeros(1, 3, 2176, 3840).cuda()
    index = 0
    for x in h_space:
        for y in w_space:
            clear_img[:, :, x:x + crop_sz_h, y:y + crop_sz_w] += output_list[index]
            index += 1
    # clear_img = clear_img.astype(np.uint8)

    return clear_img


def crop_fhd(img, args):
    img = tensor2im(torch.squeeze(img))
    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))
    crop_sz_h = 512
    crop_sz_w = 480
    h_space = [0, 512]
    w_space = [0, 480, 960, 1440]
    index = 0
    moire_list = []
    score_list = []
    for x in h_space:
        for y in w_space:
            if n_channels == 2:
                crop_img = img[x:x + crop_sz_h, y:y + crop_sz_w]
            else:
                crop_img = img[x:x + crop_sz_h, y:y + crop_sz_w, :]
            moire_list.append(crop_img)

    return moire_list, h_space, w_space, crop_sz_h, crop_sz_w


def combine_fhd(output_list, h_space, w_space, crop_sz_h, crop_sz_w, args):

    clear_img = torch.zeros(1, 3, 1024, 1920).cuda()
    index = 0
    for x in h_space:
        for y in w_space:
            clear_img[:, :, x:x + crop_sz_h, y:y + crop_sz_w] += output_list[index]
            index += 1
    # clear_img = clear_img.astype(np.uint8)

    return clear_img

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


def tensor2im(input_image, imtype=np.uint8):

    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.detach() # true
    else:
        return input_image

    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.swapaxes(image_numpy, -3,-2)
    image_numpy = np.swapaxes(image_numpy, -2,-1)

    image_numpy = image_numpy * 255


    return image_numpy.astype(imtype)


def PSNR(original, contrast):

    original = original*255.
    contrast = contrast*255.

    mse = np.mean((original - contrast) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return PSNR


def save_single_image(img, img_path):
    if np.shape(img)[-1] ==1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img * 255
    cv2.imwrite(img_path, img)


def pixel_unshuffle(batch_input, shuffle_scale = 2, device=torch.device('cuda')):
    batch_size = batch_input.shape[0]
    num_channels = batch_input.shape[1]
    height = batch_input.shape[2]
    width = batch_input.shape[3]

    conv1 = nn.Conv2d(1, 1, 2, 2, bias=False)
    conv1 = conv1.to(device)
    conv1.weight.data = torch.from_numpy(np.array([[1, 0],
                                                    [0, 0]], dtype='float32').reshape((1, 1, 2, 2))).to(device)

    conv2 = nn.Conv2d(1, 1, 2, 2, bias=False)
    conv2 = conv2.to(device)
    conv2.weight.data = torch.from_numpy(np.array([[0, 1],
                                                    [0, 0]], dtype='float32').reshape((1, 1, 2, 2))).to(device)
    conv3 = nn.Conv2d(1, 1, 2, 2, bias=False)
    conv3 = conv3.to(device)
    conv3.weight.data = torch.from_numpy(np.array([[0, 0],
                                                    [1, 0]], dtype='float32').reshape((1, 1, 2, 2))).to(device)
    conv4 = nn.Conv2d(1, 1, 2, 2, bias=False)
    conv4 = conv4.to(device)
    conv4.weight.data = torch.from_numpy(np.array([[0, 0],
                                                    [0, 1]], dtype='float32').reshape((1, 1, 2, 2))).to(device)
    Unshuffle = torch.ones((batch_size, 4, height//2, width//2), requires_grad=False).to(device)

    for i in range(num_channels):
        each_channel = batch_input[:, i:i+1, :, :]
        first_channel = conv1(each_channel)
        second_channel = conv2(each_channel)
        third_channel = conv3(each_channel)
        fourth_channel = conv4(each_channel)
        result = torch.cat((first_channel, second_channel, third_channel, fourth_channel), dim=1)
        Unshuffle = torch.cat((Unshuffle, result), dim=1)

    Unshuffle = Unshuffle[:, 4:, :, :]
    return Unshuffle.detach()


def default_loader(path):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    region = img.crop((1+int(0.15*w), 1+int(0.15*h), int(0.85*w), int(0.85*h)))
    return region


def calc_pasnr_from_folder(src_path, dst_path):
    src_image_name = os.listdir(src_path)
    dst_image_name = os.listdir(dst_path)
    image_label = ['_'.join(i.split("_")[:-1]) for i in src_image_name]
    num_image = len(src_image_name)
    psnr = 0
    for ii, label in tqdm(enumerate(image_label)):
        src = os.path.join(src_path, "{}_source.png".format(label))
        dst = os.path.join(dst_path, "{}_target.png".format(label))
        src_image = default_loader(src)
        dst_image = default_loader(dst)

        single_psnr = colour.utilities.metric_psnr(src_image, dst_image, 255)
        psnr += single_psnr

    psnr /= num_image
    return psnr


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[0] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[i], img2[i]))
            return np.array(ssims).mean()
        elif img1.shape[0] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


reconstruction_function = nn.MSELoss(size_average=False)
def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD

def get_color(img):
	img = cv2.resize(img,(256,256))
	B = img[:,:,0]
	G = img[:,:,1]
	R = img[:,:,2]
	rg = torch.Tensor(R - G)
	yb = torch.Tensor(0.5*(R+G) - B)

	dev_rg = torch.std(rg)
	dev_yb = torch.std(yb)
	mean_rg = torch.mean(rg)
	mean_yb = torch.mean(yb)
	dev = torch.sqrt(dev_rg*dev_rg + dev_yb*dev_yb)
	mean = torch.sqrt(mean_rg*mean_rg + mean_yb*mean_yb)
	M = dev + 0.3*mean
	return M

def laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplac = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    mask_img = cv2.convertScaleAbs(laplac)
    return torch.mean(torch.Tensor(mask_img))

def im_score(image):
	color = get_color(image)
	edge = laplacian(image)
	im_score = color*edge
	return im_score


class CosineAnnealingWarmRestarts(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)
    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min

        self.T_cur = 0 if last_epoch < 0 else last_epoch
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Step could be called after every batch update
        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         inputs, labels = sample['inputs'], sample['labels']
            >>>         optimizer.zero_grad()
            >>>         outputs = net(inputs)
            >>>         loss = criterion(outputs, labels)
            >>>         loss.backward()
            >>>         optimizer.step()
            >>>         scheduler.step(epoch + i / iters)
        This function can be called in an interleaved way.
        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
        """
        if epoch is None and self.last_epoch < 0:
            epoch = 0
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


