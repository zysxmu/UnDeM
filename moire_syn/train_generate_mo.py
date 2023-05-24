from __future__ import print_function
import time
import random
import argparse
import itertools

import numpy as np


import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils import ReplayBuffer
from utils import LambdaLR
from utils import createpath
from model import define_models
from dataset_mo import FHDMI_dataset, UHDM_dataset


def get_dataloader(opt):

    if opt.dataset == 'fhdmi':
        train_dataset = FHDMI_dataset
    elif opt.dataset == 'uhdm':
        train_dataset = UHDM_dataset
    else:
        raise ValueError('no this dataset choise')

    Moiredata_train = train_dataset(opt.traindata_path, patch_size=opt.patch_size)
    train_dataloader = DataLoader(Moiredata_train,
                                         batch_size=opt.batchsize,
                                         shuffle=True,
                                         num_workers=opt.num_worker,
                                         drop_last=True)

    return train_dataloader

def get_losses():
    criterion_GAN = torch.nn.MSELoss()  # lsgan
    criterion_identity = torch.nn.L1Loss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    return criterion_GAN, criterion_identity, kl_loss


def get_optimizer_LRsch(opt, netG_color, netG_mo, netD_color, netD_mo, encoder_mo, encoder_content):
    optimizer_G = torch.optim.Adam(itertools.chain(
        netG_color.parameters(), netG_mo.parameters(), encoder_mo.parameters(), encoder_content.parameters()),
                                   lr=opt.lr, betas=(0.5, 0.999))

    optimizer_D = torch.optim.Adam(itertools.chain(
        netD_color.parameters(), netD_mo.parameters()),
                                   lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                       opt.decay_epoch).step)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                       opt.decay_epoch).step)
    return optimizer_G, optimizer_D, lr_scheduler_G, lr_scheduler_D


def get_tensors_buffer(opt):
    Tensor = torch.cuda.FloatTensor
    target_real = Variable(Tensor(opt.batchsize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchsize).fill_(0.0), requires_grad=False)

    fake_color_buffer = ReplayBuffer(max_size=200)
    fake_mo_buffer = ReplayBuffer(max_size=200)

    return fake_color_buffer, fake_mo_buffer, target_real, target_fake




parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchsize', type=int, default=1, help='size of the batches')
parser.add_argument('--num_worker', type=int, default=8,
                    help='number of workers')

parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=50,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--iter_loss', type=int, default=1000, help='average loss for n iterations')
parser.add_argument('--seed', type=int, default=628)


#*********** dataset
parser.add_argument('--traindata_path', type=str, # set the datapath of patches class1 here
                    default= '/datapath/UHDM/class1')
parser.add_argument('--savefilename', type=str, #train10000
                    default= '',    help='vit_patches_size, default is 16')
parser.add_argument('--dataset',default='uhdm',help='set dataset')
parser.add_argument('--patch_size',default=128,type=int,help='set patch size') # set the patch_size!


opt = parser.parse_args()

torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
random.seed(opt.seed)
np.random.seed(opt.seed)

createpath(opt)


print(opt)
open(opt.log_path, 'a').write(str(opt) + '\n')



###### Definition of variables ######
# Networks
netG_color, netG_mo, netD_color, netD_mo, encoder_mo, encoder_content = define_models()

# Losses
criterion_GAN, criterion_identity, kl_loss = get_losses()

# Optimizers & LR schedulers
optimizer_G, optimizer_D, lr_scheduler_G, lr_scheduler_D = \
    get_optimizer_LRsch(opt, netG_color, netG_mo, netD_color, netD_mo, encoder_mo, encoder_content)

# Inputs & targets memory allocation
fake_color_buffer, fake_mo_buffer, target_real, target_fake = get_tensors_buffer(opt)

# Dataset loader
dataloader = get_dataloader(opt)

print('len dataloader', len(dataloader))
open(opt.log_path, 'w').write(str(opt) + '\n\n')



G_loss_G_temp = 0
G_loss_GAN_color_temp = 0
G_loss_GAN_mo_temp = 0
G_loss_mo_feat_temp = 0
G_loss_content_feat_temp = 0

D_loss_D_temp = 0
loss_netD_mo_temp = 0
###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    time_start = time.time()
    for i, batch in enumerate(dataloader):
        # get data
        mo, clean, mo2, clean2 = batch

        mo, clean, mo2, clean2 = mo.cuda(), clean.cuda(), mo2.cuda(), clean2.cuda()
        mo1, clean1 = mo, clean

        #### feature
        real_mo_feat = encoder_mo(mo)
        real_content = encoder_content(clean)

        fake_mo = netG_mo(real_mo_feat, clean)

        fake_mo_feat = encoder_mo(fake_mo)
        fake_content = encoder_content(fake_mo)


        ### fake fake_mo gan loss
        pred_fake_mo = netD_mo(fake_mo)
        loss_GAN_mo = criterion_GAN(pred_fake_mo, target_real)

        # feature identity
        loss_mo_feat = criterion_identity(real_mo_feat, fake_mo_feat)
        loss_content_feat = criterion_identity(real_content, fake_content)

        loss_G = loss_GAN_mo + loss_mo_feat + loss_content_feat
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        G_loss_G_temp += loss_G.item()
        G_loss_GAN_mo_temp += loss_GAN_mo.item()
        G_loss_mo_feat_temp += loss_mo_feat.item()
        G_loss_content_feat_temp += loss_content_feat.item()


        # ----- #
        # Real loss
        pred_1 = netD_mo(mo1)
        pred_2 = netD_mo(mo2)

        pred_3 = netD_mo(fake_mo_buffer.push_and_pop(fake_mo.detach()))
        loss_netD_mo = criterion_GAN(pred_1, target_real) + criterion_GAN(pred_2, target_real) + \
                       criterion_GAN(pred_3, target_fake)
        loss_netD_mo = loss_netD_mo / 3
        loss_netD_mo_temp += loss_netD_mo.item()

        # Total loss
        loss_D = loss_netD_mo
        D_loss_D_temp += loss_D.item()

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        if (i+1) % opt.iter_loss == 0:
            time_end = time.time()
            time_sum = time_end - time_start
            time_log = '%d iteration time: %.3f' % (opt.iter_loss, time_sum)
            print(time_log)
            open(opt.log_path, 'a').write(time_log + '\n')

            log = 'Epoch: %d, [iter %d], [loss_G %.5f]' \
                  '[loss_GAN_mo %.5f],' \
                  '[loss_mo_feat %.5f]' \
                  '[loss_content_feat %.5f]\n' % \
                  (epoch + 1, i+1, loss_G, loss_GAN_mo,
                   loss_mo_feat, loss_content_feat
                   )
            print(log)
            open(opt.log_path, 'a').write(log + '\n')

            log = 'Epoch: %d, [iter %d], [loss_D %.5f], [loss_netD_mo %.5f] \n' % \
                  (epoch + 1, i+1, loss_D, loss_netD_mo)

            print(log)
            open(opt.log_path, 'a').write(log + '\n')

            G_loss_G_temp /= opt.iter_loss
            G_loss_GAN_color_temp /= opt.iter_loss
            G_loss_GAN_mo_temp /= opt.iter_loss
            G_loss_mo_feat_temp /= opt.iter_loss
            G_loss_content_feat_temp /= opt.iter_loss

            D_loss_D_temp /= opt.iter_loss
            loss_netD_mo_temp /= opt.iter_loss

            avg_log = '[the last %d iters], [G_loss_G_temp %.5f] [G_loss_GAN_color_temp %.5f]' \
                      '[G_loss_GAN_mo_temp %.5f], [G_loss_mo_feat_temp %.5f], ' \
                      '[G_loss_content_feat_temp %.5f]\n' \
                      '[D_loss_D_temp %.5f]' \
                      '[loss_netD_mo_temp %.5f], ' \
                      % (opt.iter_loss, G_loss_G_temp,
                         G_loss_GAN_color_temp,
                         G_loss_GAN_mo_temp,
                         G_loss_mo_feat_temp,
                         G_loss_content_feat_temp,
                         D_loss_D_temp,
                         loss_netD_mo_temp)
            print(avg_log)
            open(opt.log_path, 'a').write(avg_log + '\n')

            G_loss_G_temp = 0
            G_loss_GAN_color_temp = 0
            G_loss_GAN_mo_temp = 0
            G_loss_mo_feat_temp = 0
            G_loss_content_feat_temp = 0

            D_loss_D_temp = 0
            loss_netD_mo_temp = 0
            time_start = time.time()

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()


    if (epoch > opt.n_epochs - 10) or (epoch % 10 == 0):

        print('save!', ('ckpt/' + opt.savefilename + '/ep%d_netG_1_%d.pth' % ( opt.n_epochs, epoch + 1)))
        torch.save(netG_mo.state_dict(),
                   ('ckpt/' + opt.savefilename + '/ep%d_netG_mo_%d.pth' % (opt.n_epochs, epoch + 1)))
        torch.save(encoder_mo.state_dict(),
                   ('ckpt/' + opt.savefilename + '/ep%d_encoder_mo_%d.pth' % (opt.n_epochs, epoch + 1)))
        torch.save(encoder_content.state_dict(),
                   ('ckpt/' + opt.savefilename + '/ep%d_encoder_content_%d.pth' % (opt.n_epochs, epoch + 1)))
