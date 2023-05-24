import os
import time
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from Util.util_collections import laplacian_diff, tensor2im, save_single_image, PSNR, Time2Str, setup_logging, \
    CosineAnnealingWarmRestarts, img_pad, crop, combine
from Net.LossNet import L1_LOSS, L1_Advanced_Sobel_Loss, L1_LOSS_Noreduce, L1_Advanced_Sobel_Loss_Noreduce, \
    multi_VGGPerceptualLoss
from dataset.dataset_weeksup import FHDMI_dataset, FHDMI_dataset_test, UHDM_dataset, UHDM_dataset_test
from torchnet import meter
from skimage.metrics import peak_signal_noise_ratio
import math
import logging
import random
from model import define_models
import cv2
import torchvision


def log(*args):
    args_list = map(str, args)
    tmp = ''.join(args_list)
    logging.info(tmp)


def load_model(netG_mo_class, encoder_mo_class, path):
    path_1 = os.path.join(path, 'seed628ep100_netG_mo_71.pth')
    print('load: ' + path_1)
    netG_mo_class.load_state_dict(torch.load(path_1))

    path_1 = os.path.join(path, 'seed628ep100_encoder_mo_71.pth')
    print('load: ' + path_1)
    encoder_mo_class.load_state_dict(torch.load(path_1))

    return netG_mo_class, encoder_mo_class


def get_dataloader(args, traindata_path):
    if args.dataset == 'fhdmi':
        train_dataset = FHDMI_dataset
        test_dataset = FHDMI_dataset_test
    elif args.dataset == 'uhdm':
        train_dataset = UHDM_dataset
        test_dataset = UHDM_dataset_test
    else:
        raise ValueError('no this dataset choise')

    Moiredata_train = train_dataset(traindata_path, patch_size=args.patch_size)
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
        if args.arch == 'ESDNet-L' or args.arch == 'ESDNet':
            if args.patch_size == 384:
                netG_mo_class1, encoder_mo_class1 = load_model(netG_mo_class1, encoder_mo_class1,
                                                               os.path.join(args.generation_model_path,
                                                                            'fhdmi_fake_class1_' + str(392)))
                netG_mo_class2, encoder_mo_class2 = load_model(netG_mo_class2, encoder_mo_class2,
                                                               os.path.join(args.generation_model_path,
                                                                            'fhdmi_fake_class2_' + str(392)))
                netG_mo_class3, encoder_mo_class3 = load_model(netG_mo_class3, encoder_mo_class3,
                                                               os.path.join(args.generation_model_path,
                                                                            'fhdmi_fake_class3_' + str(392)))
                netG_mo_class4, encoder_mo_class4 = load_model(netG_mo_class4, encoder_mo_class4,
                                                               os.path.join(args.generation_model_path,
                                                                            'fhdmi_fake_class4_' + str(392)))
            else:
                netG_mo_class1, encoder_mo_class1 = load_model(netG_mo_class1, encoder_mo_class1,
                                                               os.path.join(args.generation_model_path,
                                                                            'fhdmi_fake_class1_' + str(
                                                                                args.patch_size)))
                netG_mo_class2, encoder_mo_class2 = load_model(netG_mo_class2, encoder_mo_class2,
                                                               os.path.join(args.generation_model_path,
                                                                            'fhdmi_fake_class2_' + str(
                                                                                args.patch_size)))
                netG_mo_class3, encoder_mo_class3 = load_model(netG_mo_class3, encoder_mo_class3,
                                                               os.path.join(args.generation_model_path,
                                                                            'fhdmi_fake_class3_' + str(
                                                                                args.patch_size)))
                netG_mo_class4, encoder_mo_class4 = load_model(netG_mo_class4, encoder_mo_class4,
                                                               os.path.join(args.generation_model_path,
                                                                            'fhdmi_fake_class4_' + str(
                                                                                args.patch_size)))

        else:
            netG_mo_class1, encoder_mo_class1 = load_model(netG_mo_class1, encoder_mo_class1,
                                                           os.path.join(args.generation_model_path,
                                                                        'fhdmi_fake_class1_' + str(args.patch_size)))
            netG_mo_class2, encoder_mo_class2 = load_model(netG_mo_class2, encoder_mo_class2,
                                                           os.path.join(args.generation_model_path,
                                                                        'fhdmi_fake_class2_' + str(args.patch_size)))
            netG_mo_class3, encoder_mo_class3 = load_model(netG_mo_class3, encoder_mo_class3,
                                                           os.path.join(args.generation_model_path,
                                                                        'fhdmi_fake_class3_' + str(args.patch_size)))
            netG_mo_class4, encoder_mo_class4 = load_model(netG_mo_class4, encoder_mo_class4,
                                                           os.path.join(args.generation_model_path,
                                                                        'fhdmi_fake_class4_' + str(args.patch_size)))
    elif args.dataset == 'uhdm':
        if args.arch == 'ESDNet-L' or args.arch == 'ESDNet':
            if args.patch_size == 384:
                netG_mo_class1, encoder_mo_class1 = load_model(netG_mo_class1, encoder_mo_class1,
                                                               os.path.join(args.generation_model_path,
                                                                            'uhdm_fake_class1_' + str(392)))
                netG_mo_class2, encoder_mo_class2 = load_model(netG_mo_class2, encoder_mo_class2,
                                                               os.path.join(args.generation_model_path,
                                                                            'uhdm_fake_class2_' + str(392)))
                netG_mo_class3, encoder_mo_class3 = load_model(netG_mo_class3, encoder_mo_class3,
                                                               os.path.join(args.generation_model_path,
                                                                            'uhdm_fake_class3_' + str(392)))
                netG_mo_class4, encoder_mo_class4 = load_model(netG_mo_class4, encoder_mo_class4,
                                                               os.path.join(args.generation_model_path,
                                                                            'uhdm_fake_class4_' + str(392)))
            else:
                netG_mo_class1, encoder_mo_class1 = load_model(netG_mo_class1, encoder_mo_class1,
                                                               os.path.join(args.generation_model_path,
                                                                            'uhdm_fake_class1_' + str(
                                                                                args.patch_size)))
                netG_mo_class2, encoder_mo_class2 = load_model(netG_mo_class2, encoder_mo_class2,
                                                               os.path.join(args.generation_model_path,
                                                                            'uhdm_fake_class2_' + str(
                                                                                args.patch_size)))
                netG_mo_class3, encoder_mo_class3 = load_model(netG_mo_class3, encoder_mo_class3,
                                                               os.path.join(args.generation_model_path,
                                                                            'uhdm_fake_class3_' + str(
                                                                                args.patch_size)))
                netG_mo_class4, encoder_mo_class4 = load_model(netG_mo_class4, encoder_mo_class4,
                                                               os.path.join(args.generation_model_path,
                                                                            'uhdm_fake_class4_' + str(
                                                                                args.patch_size)))
        else:
            netG_mo_class1, encoder_mo_class1 = load_model(netG_mo_class1, encoder_mo_class1,
                                                           os.path.join(args.generation_model_path,
                                                                        'uhdm_fake_class1_' + str(args.patch_size)))
            netG_mo_class2, encoder_mo_class2 = load_model(netG_mo_class2, encoder_mo_class2,
                                                           os.path.join(args.generation_model_path,
                                                                        'uhdm_fake_class2_' + str(args.patch_size)))
            netG_mo_class3, encoder_mo_class3 = load_model(netG_mo_class3, encoder_mo_class3,
                                                           os.path.join(args.generation_model_path,
                                                                        'uhdm_fake_class3_' + str(args.patch_size)))
            netG_mo_class4, encoder_mo_class4 = load_model(netG_mo_class4, encoder_mo_class4,
                                                           os.path.join(args.generation_model_path,
                                                                        'uhdm_fake_class4_' + str(args.patch_size)))
    return  netG_mo_class1, netG_mo_class2, netG_mo_class3, netG_mo_class4, \
            encoder_mo_class1, encoder_mo_class2, encoder_mo_class3, encoder_mo_class4

def train(args, model):
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    args.save_prefix = args.save_prefix + '/' + args.arch + '_' + args.dataset + 'stage1_patch' + str(
        args.patch_size) + '_e71_denoiseClasswise50-20_onlyclass1-4_noDenoise'
    if not os.path.exists(args.save_prefix):    os.makedirs(args.save_prefix)
    setup_logging(os.path.join(args.save_prefix, 'log.txt'))
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(args.save_prefix, 'tflog'), comment=args.note)
        log(f'tensorboard path = \t\t\t{writer.log_dir}')
    log('torch devices = \t\t\t', args.device)
    log('save_path = \t\t\t\t', args.save_prefix)
    log(f'name: {args.name} note: {args.note}')

    args.pthfoler = os.path.join(args.save_prefix, '1pth_folder/')
    args.psnrfolder = os.path.join(args.save_prefix, '1psnr_folder/')
    if not os.path.exists(args.pthfoler):   os.makedirs(args.pthfoler)
    if not os.path.exists(args.psnrfolder):   os.makedirs(args.psnrfolder)

    if args.dataset == 'fhdmi':
        test_dataset = FHDMI_dataset_test
    elif args.dataset == 'uhdm':
        test_dataset = UHDM_dataset_test
    else:
        raise ValueError('no this dataset choise')

    # split dataset into patches
    Moiredata_test = test_dataset(args.testdata_path)
    test_dataloader = DataLoader(Moiredata_test,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=args.num_worker,
                                 drop_last=False)

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

    train_dataloader_class1 = get_dataloader(args, os.path.join(args.traindata_path, 'class1/'))
    train_dataloader_class2 = get_dataloader(args, os.path.join(args.traindata_path, 'class2/'))
    train_dataloader_class3 = get_dataloader(args, os.path.join(args.traindata_path, 'class3/'))
    train_dataloader_class4 = get_dataloader(args, os.path.join(args.traindata_path, 'class4/'))

    lr = args.lr
    last_epoch = 0
    if args.arch == 'ESDNet-L' or args.arch == 'ESDNet':
        optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': lr}], betas=(0.9, 0.999))
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=0.000001, last_epoch=-1)
    else:
        optimizer = optim.Adam(params=model.parameters(), lr=lr)

    list_psnr_output = []
    list_loss_output = []

    model = nn.DataParallel(model)
    encoder_mo_class1 = nn.DataParallel(encoder_mo_class1)
    encoder_mo_class2 = nn.DataParallel(encoder_mo_class2)
    encoder_mo_class3 = nn.DataParallel(encoder_mo_class3)
    encoder_mo_class4 = nn.DataParallel(encoder_mo_class4)
    netG_mo_class1 = nn.DataParallel(netG_mo_class1)
    netG_mo_class2 = nn.DataParallel(netG_mo_class2)
    netG_mo_class3 = nn.DataParallel(netG_mo_class3)
    netG_mo_class4 = nn.DataParallel(netG_mo_class4)
    if len(args.resume) > 0:
        log('load:')
        log(args.resume)
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt)

    if args.Train_pretrained_path:
        checkpoint = torch.load(args.Train_pretrained_path)
        model.load_state_dict(checkpoint['model'])
        last_epoch = checkpoint["epoch"]
        optimizer_state = checkpoint["optimizer"]
        optimizer.load_state_dict(optimizer_state)
        lr = checkpoint['lr']
        list_psnr_output = checkpoint['list_psnr_output']
        list_loss_output = checkpoint['list_loss_output']

    model = model.cuda()

    model.train()

    if args.arch == 'ESDNet-L' or args.arch == 'ESDNet':
        # loss_fn = multi_VGGPerceptualLoss(lam=1, lam_p=1).cuda()
        loss_fn = multi_VGGPerceptualLoss(lam=1, lam_p=1, reduce=False).cuda()
        loss_fn = nn.DataParallel(loss_fn)
    else:
        # criterion_l1 = L1_LOSS()
        # criterion_advanced_sobel_l1 = L1_Advanced_Sobel_Loss()
        criterion_l1 = L1_LOSS_Noreduce()
        criterion_advanced_sobel_l1 = L1_Advanced_Sobel_Loss_Noreduce()

    psnr_meter = meter.AverageValueMeter()
    Loss_meter1 = meter.AverageValueMeter()
    Loss_meter2 = meter.AverageValueMeter()
    Loss_meter3 = meter.AverageValueMeter()
    Loss_meter4 = meter.AverageValueMeter()

    if args.arch == 'MBCNN' and args.patch_size == 384:
        transform = torchvision.transforms.Resize((392, 392))

    for epoch in range(args.max_epoch):
        train_loader_1 = iter(train_dataloader_class1)
        train_loader_2 = iter(train_dataloader_class2)
        train_loader_3 = iter(train_dataloader_class3)
        train_loader_4 = iter(train_dataloader_class4)

        if args.dataset == 'fhdmi':
            batch_num = int(len(train_dataloader_class1) // 2)
        if args.dataset == 'uhdm':
            batch_num = int(len(train_dataloader_class1) // 1.5)

        if epoch < last_epoch:
            continue
        log('\nepoch = {} / {}'.format(epoch + 1, args.max_epoch))
        start = time.time()

        Loss_meter1.reset()
        Loss_meter2.reset()
        Loss_meter3.reset()
        Loss_meter4.reset()
        psnr_meter.reset()

        time_start = time.time()
        for ii in range(0, batch_num):

            # index = math.floor(ii/3)
            c = random.randint(0, 3)
            # c = random.randint(2, 3)
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

            if args.arch == 'MBCNN' and args.patch_size == 384:
                fake_mo = transform(fake_mo)
                clear1 = transform(clear1)

            # class-wise noise cancel
            a = laplacian_diff(clear1, fake_mo)
            if args.dataset == 'fhdmi':
                if args.patch_size == 192:
                    if c == 0:
                        t = a > 5.587634 # 50
                    elif c == 1:
                        t = a > 10.194368 # 40
                    elif c == 2:
                        t = a > 18.6177 # 30
                    elif c == 3:
                        t = a > 43.1079 # 20
                        # t = a > 55.5370  # 10
                elif args.patch_size == 384 or args.patch_size == 392:
                    if c == 0:
                        t = a > 5.572998 # 50
                    elif c == 1:
                        t = a > 7.421407 # 40
                    elif c == 2:
                        t = a > 15.617778 # 30
                    elif c == 3:
                        t = a > 31.256857 # 20
                else:
                    raise ValueError('no this patch_size choise for fhdmi')
            elif args.dataset == 'uhdm':
                if args.patch_size == 192:
                    if c == 0:
                        t = a > 13.6871 # 50
                    elif c == 1:
                        t = a > 20.1079
                    elif c == 2:
                        t = a > 38.4620
                    elif c == 3:
                        t = a > 31.8185 # 20
                elif args.patch_size == 384 or args.patch_size == 392:
                    if c == 0:
                        t = a > 16.0618  # 50
                    elif c == 1:
                        t = a > 24.4521
                    elif c == 2:
                        t = a > 46.1461
                    elif c == 3:
                        t = a > 131.7934  # 20
                elif args.patch_size == 768:
                    if c == 0:
                        t = a > 15.1093  # 50
                    elif c == 1:
                        t = a > 22.4396
                    elif c == 2:
                        t = a > 51.4990
                    elif c == 3:
                        t = a > 199.3884  # 20
                else:
                    raise ValueError('no this patch_size choise for uhdm')
            else:
                raise ValueError('no this patch_size choise for dataset')
            t = t.cuda()

            if args.arch == 'MBCNN':
                output3, output2, output1 = model(fake_mo)  # 32,1,256,256 = 32,1,256,256
                # output1 = model(moires)
                Loss_l1 = criterion_l1(output1, clear1)
                Loss_advanced_sobel_l1 = criterion_advanced_sobel_l1(output1, clear1)
                Loss_l12 = criterion_l1(output2, clear2)
                Loss_advanced_sobel_l12 = criterion_advanced_sobel_l1(output2, clear2)
                Loss_l13 = criterion_l1(output3, clear3)
                Loss_advanced_sobel_l13 = criterion_advanced_sobel_l1(output3, clear3)

                Loss1 = Loss_l1 + (0.25) * Loss_advanced_sobel_l1
                Loss2 = Loss_l12 + (0.25) * Loss_advanced_sobel_l12
                Loss3 = Loss_l13 + (0.25) * Loss_advanced_sobel_l13

                Loss1 = torch.mean(Loss1 * ~t)
                Loss2 = torch.mean(Loss2 * ~t)
                Loss3 = torch.mean(Loss3 * ~t)

                loss = Loss1 + Loss2 + Loss3

                # loss_check1 = Loss1
                # loss_check2 = Loss_l1
                # loss_check3 = (0.25) * Loss_advanced_sobel_l1

                loss_check1 = Loss1
                loss_check2 = torch.mean(Loss_l1 * ~t)
                loss_check3 = torch.mean((0.25) * Loss_advanced_sobel_l1 * ~t)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                moires = tensor2im(moires)
                output1 = tensor2im(output1)
                clear1 = tensor2im(clear1)

                psnr = peak_signal_noise_ratio(output1, clear1)
                psnr_meter.add(psnr)
                Loss_meter1.add(loss.item())
                Loss_meter2.add(loss_check1.item())
                Loss_meter3.add(loss_check2.item())
                Loss_meter4.add(loss_check3.item())

            elif args.arch == 'ESDNet-L' or args.arch == 'ESDNet':
                out_1, out_2, out_3 = model(fake_mo)
                loss = loss_fn(out_1, out_2, out_3, clear1)

                loss = torch.mean(loss * ~t)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                moires = tensor2im(moires)
                output1 = tensor2im(out_1)
                clear1 = tensor2im(clear1)

                psnr = peak_signal_noise_ratio(output1, clear1)
                psnr_meter.add(psnr)
                Loss_meter1.add(loss.item())
                Loss_meter2.add(loss.item())
                Loss_meter3.add(loss.item())
            else:
                output1 = model(moires)  # 32,1,256,256 = 32,1,256,256\

                Loss_l1 = criterion_l1(output1, clear1)

                Loss_l1 = torch.mean(Loss_l1 * ~t)

                Loss1 = Loss_l1
                loss = Loss1



                loss_check1 = Loss1
                loss_check2 = Loss_l1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                moires = tensor2im(moires)

                output1 = tensor2im(output1)
                clear1 = tensor2im(clear1)

                psnr = peak_signal_noise_ratio(output1, clear1)
                psnr_meter.add(psnr)
                Loss_meter1.add(loss.item())
                Loss_meter2.add(loss_check1.item())
                Loss_meter3.add(loss_check2.item())
                # break

            if ii % 100 == 0:
                time_end = time.time()
                time_sum = time_end - time_start
                time_log = '%d iteration time: %.3f' % (100, time_sum)
                time_start = time.time()
                print(time_log)
                log('iter: {} Total iter: {} \ttraining set : \tPSNR = {:f}\t loss = {:f}\t Loss1(scale) = {:f} \t Loss_L1 = {:f} + Loss_sobel = {:f},\t '
                    .format(ii, batch_num, psnr_meter.value()[0], Loss_meter1.value()[0], Loss_meter2.value()[0],
                            Loss_meter3.value()[0], Loss_meter4.value()[0]))

        if args.dataset == 'uhdm':
            if epoch % 1 == 0:
                psnr_output, loss_output1, loss_output2, loss_output3, loss_output4 = val(model, test_dataloader, epoch,
                                                                                          args)
                log('Test set : \tloss = {:0.4f} \t Loss_1 = {:0.4f} \t Loss_L1 = {:0.4f} \t Loss_ASL = {:0.4f}'.format(
                    loss_output1, loss_output2, loss_output3, loss_output4))
                log('Test set : \t' + '\033[30m \033[43m' + 'PSNR = {:0.4f}'.format(
                    psnr_output) + '\033[0m' + '\tbest PSNR ={:0.4f}'.format(args.bestperformance))
        else:
            psnr_output, loss_output1, loss_output2, loss_output3, loss_output4 = val(model, test_dataloader, epoch,
                                                                                      args)
            log('Test set : \tloss = {:0.4f} \t Loss_1 = {:0.4f} \t Loss_L1 = {:0.4f} \t Loss_ASL = {:0.4f}'.format(
                loss_output1, loss_output2, loss_output3, loss_output4))
            log('Test set : \t' + '\033[30m \033[43m' + 'PSNR = {:0.4f}'.format(
                psnr_output) + '\033[0m' + '\tbest PSNR ={:0.4f}'.format(args.bestperformance))

        list_psnr_output.append(round(psnr_output, 5))
        list_loss_output.append(round(loss_output1, 5))
        if args.arch == 'ESDNet-L' or args.arch == 'ESDNet':
            scheduler.step()
        else:
            if epoch > 5:
                list_tmp = list_loss_output[-5:]
                for j in range(4):
                    sub = 10 * (math.log10((list_tmp[j] / list_tmp[j + 1])))
                    if sub > 0.001: break
                    if j == 3:
                        log('\033[30m \033[41m' + 'LR was Decreased!!!{:} > {:}\t\t\t\t\t\t\t\t\t'.format(lr,
                                                                                                          lr / 2) + '\033[0m')
                        lr = lr * 0.5
                        for param_group in optimizer.param_groups:  param_group['lr'] = lr
                    if lr < 1e-6:                                   exit()

        if psnr_output > args.bestperformance:
            args.bestperformance = psnr_output
            file_name = args.pthfoler + 'ckpt_best.pth'
            torch.save(model.state_dict(), file_name)
            log('\033[30m \033[42m' + 'PSNR WAS UPDATED! ' + '\033[0m')

        if (epoch + 1) % args.save_every == 0 or epoch == 0:
            file_name = args.pthfoler + 'ckpt_last.pth'
            checkpoint = {'epoch': epoch + 1,
                          "optimizer": optimizer.state_dict(),
                          "model": model.state_dict(),
                          "lr": lr,
                          "list_psnr_output": list_psnr_output,
                          "list_loss_output": list_loss_output,
                          }
            torch.save(checkpoint, file_name)

            with open(args.save_prefix + "/1_PSNR_validation_set_output_psnr.txt", 'w') as f:
                f.write("psnr_output: {:}\n".format(list_psnr_output))
            with open(args.save_prefix + "/1_Loss_validation_set_output_loss.txt", 'w') as f:
                f.write("loss_output: {:}\n".format(list_loss_output))

        if epoch == (args.max_epoch - 1):
            file_name2 = args.pthfoler + '{0}_stdc_epoch{1}.pth'.format(args.name, epoch + 1)
            torch.save(model.state_dict(), file_name2)

        log('1 epoch spends:{:.2f}sec\t remain {:2d}:{:2d} hours'.format(
            (time.time() - start),
            int((args.max_epoch - epoch) * (time.time() - start) // 3600),
            int((args.max_epoch - epoch) * (time.time() - start) % 3600 / 60)))

    return "Training Finished!"


def val(model, dataloader, epoch, args):  # 맨처음 확인할때의 epoch == -1

    model.eval()

    # criterion_l2 = L2_LOSS()
    criterion_l1 = L1_LOSS()
    criterion_advanced_sobel_l1 = L1_Advanced_Sobel_Loss()

    psnr_output_meter = meter.AverageValueMeter()
    loss_meter1 = meter.AverageValueMeter()
    loss_meter2 = meter.AverageValueMeter()
    loss_meter3 = meter.AverageValueMeter()
    loss_meter4 = meter.AverageValueMeter()

    psnr_output_meter.reset()
    loss_meter1.reset()
    loss_meter2.reset()
    loss_meter3.reset()
    loss_meter4.reset()

    image_train_path_demoire = "{0}/epoch_{1}_validation_set_{2}/".format(args.save_prefix, epoch + 1, "demoire")
    if not os.path.exists(image_train_path_demoire) and (epoch + 1) % args.save_every == 0: os.makedirs(
        image_train_path_demoire)

    image_train_path_moire = "{0}/epoch_{1}_validation_set_{2}/".format(args.save_prefix, 1, "moire")
    image_train_path_clean = "{0}/epoch_{1}_validation_set_{2}/".format(args.save_prefix, 1, "clean")
    if not os.path.exists(image_train_path_moire): os.makedirs(image_train_path_moire)
    if not os.path.exists(image_train_path_clean): os.makedirs(image_train_path_clean)

    mytrans = torchvision.transforms.ToTensor()
    for ii, (val_moires, val_clears_list, labels) in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            val_moires = val_moires.to(args.device)

            if args.dataset == 'uhdm':
                # the same as UHDM code
                b, c, h, w = val_moires.size()
                # pad image such that the resolution is a multiple of 32
                w_pad = (math.ceil(w / 32) * 32 - w) // 2
                h_pad = (math.ceil(h / 32) * 32 - h) // 2
                w_odd_pad = w_pad
                h_odd_pad = h_pad
                if w % 2 == 1:
                    w_odd_pad += 1
                if h % 2 == 1:
                    h_odd_pad += 1
                val_moires = img_pad(val_moires, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)

            _, _, clear1 = val_clears_list
            clear1 = clear1.to(args.device)
            if args.arch == 'MBCNN':
                if args.dataset == 'uhdm':
                    moire_list, h_space, w_space, crop_sz_h, crop_sz_w = crop(val_moires, args)
                    output_list = []
                    for i, moire_patch in enumerate(moire_list):
                        moire_patch = torch.unsqueeze(mytrans(moire_patch), 0).to(args.device)
                        _, _, out = model(moire_patch)
                        # output_list.append(tensor2im(torch.squeeze(out)))
                        output_list.append(out)
                    output1 = combine(output_list, h_space, w_space, crop_sz_h, crop_sz_w, args)
                else:
                    output3, output2, output1 = model(val_moires)
            elif args.arch == 'ESDNet-L' or args.arch == 'ESDNet':
                output1, _, _ = model(val_moires)
            else:
                output1 = model(val_moires)
            if args.dataset == 'uhdm':
                # the same as UHDM code
                if h_pad != 0:
                    output1 = output1[:, :, h_pad:-h_odd_pad, :]
                    val_moires = val_moires[:, :, h_pad:-h_odd_pad, :]
                if w_pad != 0:
                    output1 = output1[:, :, :, w_pad:-w_odd_pad]
                    val_moires = val_moires[:, :, :, w_pad:-w_odd_pad]

        # for LR sch of MBCNN
        loss_l1 = criterion_l1(output1, clear1)
        loss_advanced_sobel_l1 = criterion_advanced_sobel_l1(output1, clear1)
        Loss1 = loss_l1 + (0.25) * loss_advanced_sobel_l1
        loss = Loss1
        loss_meter1.add(loss.item())

        val_moires = tensor2im((val_moires))  # type tensor to numpy .detach().cpu().float().numpy()
        output1 = tensor2im((output1))
        clear1 = tensor2im((clear1))

        bs = val_moires.shape[0]
        if epoch != -1:
            for jj in range(bs):
                output, clear, moire, label = output1[jj], clear1[jj], val_moires[jj], labels[jj]

                psnr_output_individual = peak_signal_noise_ratio(output, clear)
                psnr_output_meter.add(psnr_output_individual)
                psnr_input_individual = peak_signal_noise_ratio(moire, clear)
        #
        #         if (epoch + 1) % args.save_every == 0 or epoch == 0:  # 每5个epoch保存一 次
        #             img_path = "{0}/{1}_epoch:{2:04d}_demoire_PSNR:{3:.4f}_demoire.png".format(image_train_path_demoire,
        #                                                                                        label, epoch + 1,
        #                                                                                        psnr_output_individual)
                    # save_single_image(output, img_path)

                # if epoch == 0:
                #     psnr_in_gt = peak_signal_noise_ratio(moire, clear)
                #     img_path2 = "{0}/{1}_moire_{2:.4f}_moire.png".format(image_train_path_moire, label, psnr_in_gt)
                #     img_path3 = "{0}/{1}_clean_.png".format(image_train_path_clean, label)
                    # save_single_image(moire, img_path2)
                    # save_single_image(clear, img_path3)
        # break

    return psnr_output_meter.value()[0], loss_meter1.value()[0], loss_meter2.value()[0], loss_meter3.value()[0], \
           loss_meter4.value()[0]


