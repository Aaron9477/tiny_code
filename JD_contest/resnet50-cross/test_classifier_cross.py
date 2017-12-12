# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from PIL import Image

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='pig train')
parser.add_argument('data', metavar='DIR',default='/usr/JD/raw/train/train_set/',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
# parser.add_argument('-s', '--sequence', default=1, type=int,
#                     metavar='N', help='the sequence of cross validation (1-5)')
best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    print(model)
######################
    if args.arch.startswith('vgg'):
        mod = list(model.classifier.children())
        mod.pop()
        mod.append(torch.nn.Linear(4096, 30))
        new_classifier = torch.nn.Sequential(*mod)
        model.classifier = new_classifier
        
        
    elif args.arch.startswith('alexnet'):
        mod = list(model.classifier.children())
        mod.pop()
        mod.append(torch.nn.Linear(4096, 30))
        new_classifier = torch.nn.Sequential(*mod)
        model.classifier = new_classifier
    else:
        model.fc=torch.nn.Linear(2048, 30)
    
    print(model)
    ########################


    model.cuda()


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint


    cudnn.benchmark = True


    valdir = os.path.join(args.data, 'test_A')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform=transforms.Compose([
            transforms.Scale([224,224]),
            transforms.ToTensor(),
            normalize,
        ])
####################################################
    class_index=[1,10,11,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,29,3,30,4,5,6,7,8,9]
    result=np.zeros((3000,5,30))
    for k in range(5):
        resumename='/home/zq610/WYZ/JD_contest/process/17-11-26-2/'+str(k+1)+'_model_best.pth.tar'
        if resumename:
            if os.path.isfile(resumename):
                print("=> loading checkpoint '{}'".format(resumename))
                checkpoint = torch.load(resumename)
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                #optimizer.load_state_dict(checkpoint['optimizer'])

                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(resumename, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(resumename))

    ######################
        logpath_val='./log/test/'
        txtname='test'+str(k+1)+'.csv'
        if not os.path.exists(logpath_val):
            os.makedirs(logpath_val)
        if os.path.exists(logpath_val+txtname):
            os.remove(logpath_val+txtname)
        f_val=file(logpath_val+txtname,'a+')
    #################


        crossnum=k
        model.eval()
        images = np.array([])
        test_list = get_files(valdir)
        for i, item in enumerate(test_list):
            print('%s val:  Processing %i of %i (%s)' % (k+1,i+1, len(test_list), item))
            test_image_name=item.split('/')[-1].split('.')[0]
            test_image = transform(Image.open(item))
            input_var = torch.autograd.Variable(test_image, volatile=True).unsqueeze(0).cuda()
            output=nn.functional.softmax(model(input_var))

            f_val.writelines(test_image_name+','+str(1)+','+str('%.10f'%output.data[0][0])+'\n')
            result[i][crossnum][0]=output.data[0][0]
            f_val.writelines(test_image_name+','+str(2)+','+str('%.10f'%output.data[0][11])+'\n')
            result[i][crossnum][1]=output.data[0][11]
            f_val.writelines(test_image_name+','+str(3)+','+str('%.10f'%output.data[0][22])+'\n')
            result[i][crossnum][2]=output.data[0][22]
            f_val.writelines(test_image_name+','+str(4)+','+str('%.10f'%output.data[0][24])+'\n')
            result[i][crossnum][3]=output.data[0][24]
            f_val.writelines(test_image_name+','+str(5)+','+str('%.10f'%output.data[0][25])+'\n')
            result[i][crossnum][4]=output.data[0][25]
            f_val.writelines(test_image_name+','+str(6)+','+str('%.10f'%output.data[0][26])+'\n')
            result[i][crossnum][5]=output.data[0][26]
            f_val.writelines(test_image_name+','+str(7)+','+str('%.10f'%output.data[0][27])+'\n')
            result[i][crossnum][6]=output.data[0][27]
            f_val.writelines(test_image_name+','+str(8)+','+str('%.10f'%output.data[0][28])+'\n')
            result[i][crossnum][7]=output.data[0][28]
            f_val.writelines(test_image_name+','+str(9)+','+str('%.10f'%output.data[0][29])+'\n')
            result[i][crossnum][8]=output.data[0][29]
            f_val.writelines(test_image_name+','+str(10)+','+str('%.10f'%output.data[0][1])+'\n')
            result[i][crossnum][9]=output.data[0][1]
            f_val.writelines(test_image_name+','+str(11)+','+str('%.10f'%output.data[0][2])+'\n')
            result[i][crossnum][10]=output.data[0][2]
            f_val.writelines(test_image_name+','+str(12)+','+str('%.10f'%output.data[0][3])+'\n')
            result[i][crossnum][11]=output.data[0][3]
            f_val.writelines(test_image_name+','+str(13)+','+str('%.10f'%output.data[0][4])+'\n')
            result[i][crossnum][12]=output.data[0][4]
            f_val.writelines(test_image_name+','+str(14)+','+str('%.10f'%output.data[0][5])+'\n')
            result[i][crossnum][13]=output.data[0][5]
            f_val.writelines(test_image_name+','+str(15)+','+str('%.10f'%output.data[0][6])+'\n')
            result[i][crossnum][14]=output.data[0][6]
            f_val.writelines(test_image_name+','+str(16)+','+str('%.10f'%output.data[0][7])+'\n')
            result[i][crossnum][15]=output.data[0][7]
            f_val.writelines(test_image_name+','+str(17)+','+str('%.10f'%output.data[0][8])+'\n')
            result[i][crossnum][16]=output.data[0][8]
            f_val.writelines(test_image_name+','+str(18)+','+str('%.10f'%output.data[0][9])+'\n')
            result[i][crossnum][17]=output.data[0][9]
            f_val.writelines(test_image_name+','+str(19)+','+str('%.10f'%output.data[0][10])+'\n')
            result[i][crossnum][18]=output.data[0][10]
            f_val.writelines(test_image_name+','+str(20)+','+str('%.10f'%output.data[0][12])+'\n')
            result[i][crossnum][19]=output.data[0][12]
            f_val.writelines(test_image_name+','+str(21)+','+str('%.10f'%output.data[0][13])+'\n')
            result[i][crossnum][20]=output.data[0][13]
            f_val.writelines(test_image_name+','+str(22)+','+str('%.10f'%output.data[0][14])+'\n')
            result[i][crossnum][21]=output.data[0][14]
            f_val.writelines(test_image_name+','+str(23)+','+str('%.10f'%output.data[0][15])+'\n')
            result[i][crossnum][22]=output.data[0][15]
            f_val.writelines(test_image_name+','+str(24)+','+str('%.10f'%output.data[0][16])+'\n')
            result[i][crossnum][23]=output.data[0][16]
            f_val.writelines(test_image_name+','+str(25)+','+str('%.10f'%output.data[0][17])+'\n')
            result[i][crossnum][24]=output.data[0][17]
            f_val.writelines(test_image_name+','+str(26)+','+str('%.10f'%output.data[0][18])+'\n')
            result[i][crossnum][25]=output.data[0][18]
            f_val.writelines(test_image_name+','+str(27)+','+str('%.10f'%output.data[0][19])+'\n')
            result[i][crossnum][26]=output.data[0][19]
            f_val.writelines(test_image_name+','+str(28)+','+str('%.10f'%output.data[0][20])+'\n')
            result[i][crossnum][27]=output.data[0][20]
            f_val.writelines(test_image_name+','+str(29)+','+str('%.10f'%output.data[0][21])+'\n')
            result[i][crossnum][28]=output.data[0][21]
            f_val.writelines(test_image_name+','+str(30)+','+str('%.10f'%output.data[0][23])+'\n')
            result[i][crossnum][29]=output.data[0][23]


    ######################
    logpath_val='./log/test/'
    txtname='test_final.csv'
    if not os.path.exists(logpath_val):
        os.makedirs(logpath_val)
    if os.path.exists(logpath_val+txtname):
        os.remove(logpath_val+txtname)
    f_val=file(logpath_val+txtname,'a+')
    #################
    final_result=np.mean(result, axis=1)
    for i, item in enumerate(test_list):
        test_image_name=item.split('/')[-1].split('.')[0]
        print('====final :  Processing %i of %i (%s)' % (i+1, len(test_list), item))
        for j in range(30):
            f_val.writelines(test_image_name+','+str(j+1)+','+str('%.10f'%final_result[i][j])+'\n')
    return

    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # else:
    #     train_sampler = None

    # if args.evaluate:
    #     validate(val_loader, model, criterion)
    #     return

    # for epoch in range(args.start_epoch, args.epochs):
    #     if args.distributed:
    #         train_sampler.set_epoch(epoch)
    #     adjust_learning_rate(optimizer, epoch)

    #     # train for one epoch
    #     train(train_loader, model, criterion, optimizer, epoch)

    #     # evaluate on validation set
    #     prec1 = validate(val_loader, model, criterion)

    #     # remember best prec@1 and save checkpoint
    #     is_best = prec1 > best_prec1
    #     best_prec1 = max(prec1, best_prec1)
    #     save_checkpoint({
    #         'epoch': epoch + 1,
    #         'arch': args.arch,
    #         'state_dict': model.state_dict(),
    #         'best_prec1': best_prec1,
    #         'optimizer' : optimizer.state_dict(),
    #     }, is_best)

   

def get_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
        if os.path.isfile(os.path.join(directory, f))]

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
