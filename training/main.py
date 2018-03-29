import argparse
import os
import time
import numpy as np
import data
from importlib import import_module
import shutil
from utils import *
import sys
sys.path.append('../')

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from config_training import config as config_training
from torch import nn
import math

parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('--model', '-m', metavar='MODEL', default='base',
                    help='model')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=48, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--save-freq', default='10', type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--test', default=0, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not')
parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts')
parser.add_argument('--gpu', default='all', type=str, metavar='N',
                    help='use gpu')
parser.add_argument('--n_test', default=8, type=int, metavar='N',
                    help='number of gpu for test')

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def main():
    global args
    args = parser.parse_args()
    
    
    torch.manual_seed(0)
    torch.cuda.set_device(0)

    model = import_module(args.model)
    config, net, loss = model.get_model()
    start_epoch = args.start_epoch
    save_dir = args.save_dir
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        #if start_epoch == 0:
        #    start_epoch = checkpoint['epoch'] + 1
        #if not save_dir:
        #    save_dir = checkpoint['save_dir']
        #else:
        save_dir = os.path.join('results',save_dir)
        net.load_state_dict(checkpoint['state_dict'])
    else:
        if start_epoch == 0:
            start_epoch = 1
        if not save_dir:
            exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
            save_dir = os.path.join('results', args.model + '-' + exp_id)
        else:
            save_dir = os.path.join('results',save_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = os.path.join(save_dir,'log')
    if args.test!=1:
        sys.stdout = Logger(logfile)
        pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
        for f in pyfiles:
            shutil.copy(f,os.path.join(save_dir,f))
    n_gpu = setgpu(args.gpu)
    args.n_gpu = n_gpu
    print ("arg", args.gpu)
    print ("num_gpu",n_gpu)
    net = net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = True
    net = DataParallel(net)
    datadir = config_training['preprocess_result_path']

    print ("datadir", datadir)
    print ("pad_val", config['pad_value'])
    print ("aug type", config['augtype'])
    
    dataset = data.DataBowl3Detector(
        datadir,
        'train_luna_9.npy',
        config,
        phase = 'train')
    print ("len train_dataset", dataset.__len__())
    train_loader = DataLoader(
        dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers,
        pin_memory=True)

    dataset = data.DataBowl3Detector(
        datadir,
        'val9.npy',
        config,
        phase = 'val')
    print ("len val_dataset", dataset.__len__())

    val_loader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = args.workers,
        pin_memory=True)

    optimizer = torch.optim.SGD(
        net.parameters(),
        args.lr,
        momentum = 0.9,
        weight_decay = args.weight_decay)

    def get_lr(epoch):
        if epoch <= args.epochs * 0.5:
            lr = args.lr
        elif epoch <= args.epochs * 0.8:
            lr = 0.1 * args.lr
        else:
            lr = 0.01 * args.lr
        return lr
    
    best_val_loss = 100
    best_mal_loss = 100
    for epoch in range(start_epoch, args.epochs + 1):
        print ("epoch", epoch)
        train(train_loader, net, loss, epoch, optimizer, get_lr, args.save_freq, save_dir)
        best_val_loss, best_mal_loss =  validate(val_loader, net, loss, best_val_loss, best_mal_loss, epoch, save_dir)

def train(data_loader, net, loss, epoch, optimizer, get_lr, save_freq, save_dir):
    start_time = time.time()

    net.train()
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    loss_res = []
    for i, (data, target) in enumerate(data_loader):
        data = Variable(data.cuda(async = True))
        target = Variable(target.cuda(async = True))
        output = net(data)

        loss_output, attribute = loss(output, target.float())
        optimizer.zero_grad()
        loss_output.backward()
        optimizer.step()
        loss_res.append(loss_output.data[0])

    if epoch % args.save_freq == 0:            
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
            
        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'args': args},
            os.path.join(save_dir, '%03d.ckpt' % epoch))

    end_time = time.time()

    loss_res = np.asarray(loss_res, np.float32)
    print('Epoch %03d (lr %.5f)' % (epoch, lr))
    print('Train loss %2.4f, time %2.4f' % (np.mean(loss_res), end_time - start_time))
    print

def validate(data_loader, net, loss, best_val_loss, best_mal_loss, epoch, save_dir):
    start_time = time.time()

    net.eval()

    loss_res = []
    less_list = np.zeros((9, 10))
    malignacy_loss = []
    sphericiy_loss = []
    margin_loss = []
    spiculation_loss = []
    texture_loss = []
    calcification_loss = []
    internal_structure_loss = []
    lobulation_loss = []
    subtlety_loss = []
    for i, (data, target) in enumerate(data_loader):
        data = Variable(data.cuda(async = True), volatile = True)
        target = Variable(target.cuda(async = True), volatile = True)

        output = net(data)
        loss_output, attribute = loss(output, target.float(), train=False)

        loss_res.append(loss_output.data[0])

        for i in range(len(less_list)):
            for j in range(10):
                if attribute[i].data[0] < 0.1*(j+1):
                    less_list[i][j] = less_list[i][j] + 1


        malignacy_loss.append(attribute[0].data[0])
        sphericiy_loss.append(attribute[1].data[0])
        margin_loss.append(attribute[2].data[0])
        spiculation_loss.append(attribute[3].data[0])
        texture_loss.append(attribute[4].data[0])
        calcification_loss.append(attribute[5].data[0])
        internal_structure_loss.append(attribute[6].data[0])
        lobulation_loss.append(attribute[7].data[0])
        subtlety_loss.append(attribute[8].data[0])

    end_time = time.time()

    loss_res = np.asarray(loss_res, np.float32)
    val_loss = np.mean(loss_res)

    if ( val_loss < best_val_loss):
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'args': args},
            os.path.join(save_dir, 'best_all.ckpt'))

        best_val_loss = val_loss


        for i in range(len(less_list)):
            temp_list = np.zeros((10))
            for j in range(10):
                temp_list[j] = float(less_list[i][j]) / len(loss_res)
            print(i, 'th', 'acc less ', temp_list)

    if ( np.mean(malignacy_loss) < best_mal_loss):
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'args': args},
            os.path.join(save_dir, 'best_mal.ckpt'))

        best_mal_loss = np.mean(malignacy_loss)

        i=0
        temp_list = np.zeros((10))
        for j in range(10):
            temp_list[j] = float(less_list[i][j]) / len(loss_res)
        print(i, 'th', 'acc less ', temp_list)

    print('val loss %2.4f, time %2.4f' % (np.mean(loss_res), end_time - start_time))
    print('malignacy_loss %2.4f, sphericiy_loss %2.4f, margin_loss %2.4f' % (np.mean(malignacy_loss), np.mean(sphericiy_loss), np.mean(margin_loss)))
    print('spiculation_loss %2.4f, texture_loss %2.4f, calcification_loss %2.4f' % (np.mean(spiculation_loss), np.mean(texture_loss), np.mean(calcification_loss)))
    print('internal_structure_loss %2.4f, lobulation_loss %2.4f, subtlety_loss %2.4f' % (np.mean(internal_structure_loss), np.mean(lobulation_loss), np.mean(subtlety_loss)))

    print ('best_val_loss ', best_val_loss)

    return best_val_loss, best_mal_loss

if __name__ == '__main__':
    main()

