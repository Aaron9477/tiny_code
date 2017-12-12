#!/usr/bin/env python
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
parser.add_argument('data', metavar='DIR',default='/home/zq610/WYZ/JD_contest/wipe_out/wipe_out_small/output/',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
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


    mod = list(model.children())
    mod.pop()
    new_classifier = torch.nn.Sequential(*mod)
    model = new_classifier
    model.cuda()

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda()

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         best_prec1 = checkpoint['best_prec1']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         #optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))
######################
    # logpath_val='./log/test/'
    # txtname='test.csv'
    # if not os.path.exists(logpath_val):
    #     os.makedirs(logpath_val)
    # if os.path.exists(logpath_val+txtname):
    #     os.remove(logpath_val+txtname)
    # f_val=file(logpath_val+txtname,'a+')
#################
    testdir='/home/zq610/WYZ/JD_contest/test/test_A/'
    cudnn.benchmark = True

        
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform=transforms.Compose([
    transforms.Scale([224,224]),
    transforms.ToTensor(),
    normalize,])
####################################################
    #class_index=[1,10,11,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,29,3,30,4,5,6,7,8,9]
    model.eval()
    test_list = get_files(testdir)
    num=len(test_list)
    featuresall=np.zeros((num,2048))
    images = np.array([])
    for i, item in enumerate(test_list):
        print('Processing %i of %i (%s)' % (i+1, len(test_list), item))
        test_image_name=item.split('/')[-1].split('.')[0]
        test_image = transform(Image.open(item))
        input_var = torch.autograd.Variable(test_image, volatile=True).unsqueeze(0).cuda()##[1,3,224,224]
        output=model(input_var)
        features=output.data[0][:,:,0].t().cpu().numpy()###  feature vector: [1*2048]
        featuresall[i,:]=features
    np.save(args.data+'../test_A_resnet50.npy',featuresall)

    for classes in range(30):
        valdir = os.path.join(args.data,str(classes+1))
        test_list = get_files(valdir)
        num=len(test_list)
        featuresall=np.zeros((num,2048))
        images = np.array([])
        for i, item in enumerate(test_list):
            print('Processing %i of %i (%s)' % (i+1, len(test_list), item))
            test_image_name=item.split('/')[-1].split('.')[0]
            test_image = transform(Image.open(item))
            input_var = torch.autograd.Variable(test_image, volatile=True).unsqueeze(0).cuda()##[1,3,224,224]
            output=model(input_var)
            features=output.data[0][:,:,0].t().cpu().numpy()###  feature vector: [1*2048]
            featuresall[i,:]=features
        np.save(args.data+str(classes+1)+'_resnet50.npy',featuresall)



def get_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
        if os.path.isfile(os.path.join(directory, f))]

if __name__ == '__main__':
    main()