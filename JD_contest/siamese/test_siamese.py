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
import torch.nn.functional as F
import numpy
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
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
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


def pairwise_distance1(a,b):
    c=abs(a-b)
    d=numpy.linalg.norm(c,2)
    return d 

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

    class MyResNet(nn.Module):

        def __init__(self, model):
            super(MyResNet, self).__init__()
            self.conv1 = model.conv1
            self.bn1 = model.bn1
            self.relu = model.relu
            self.maxpool = model.maxpool
            self.layer1 = model.layer1
            self.layer2 = model.layer2
            self.layer3 = model.layer3
            self.layer4 = model.layer4
            self.avgpool = model.avgpool
            #self.Net_classifier=Net_classifier


        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            #x = x.view(x.size(0), -1)
            #x = self.Net_classifier(x)
            return x

    my_model=MyResNet(model)


    class SiameseNetwork(nn.Module):
        def __init__(self):
            super(SiameseNetwork, self).__init__()
            self.cnn1 = my_model

        def forward_once(self, x):
            output = self.cnn1(x)
            #output = output.view(output.size()[0], -1)
            #output = self.fc1(output)
            return output

        def forward(self, input1):
            output1 = self.forward_once(input1)
            #output2 = self.forward_once(input2)
            #return output1, output2
            return output1
#############################################
    this_model = SiameseNetwork().cuda()
    model = this_model.cuda()
    print(model)
    model.cuda()


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            #args.start_epoch = checkpoint['epoch']
            #best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint)
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint OK!!!!!!!!!!!!!")
            #      .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
######################
    logpath_val='./log/test/'
    txtname='test.csv'
    if not os.path.exists(logpath_val):
        os.makedirs(logpath_val)
    if os.path.exists(logpath_val+txtname):
        os.remove(logpath_val+txtname)
    f_val=file(logpath_val+txtname,'a+')
#################
    cudnn.benchmark = True


    valdir = os.path.join(args.data, 'test_B')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform=transforms.Compose([
            transforms.Scale([224,224]),
            transforms.ToTensor(),
            normalize,
        ])
####################################################
    class_index=[1,10,11,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,29,3,30,4,5,6,7,8,9]
    model.eval()
    images = np.array([])
    test_list = get_files(valdir)

    train_dir = '/home/zq610/WYZ/JD_contest/wipe_out/wipe_out_small/output/'

    for i, item in enumerate(test_list):
        print('Processing %i of %i (%s)' % (i+1, len(test_list), item))
        test_image_name=item.split('/')[-1].split('.')[0]   #file name
        test_image = transform(Image.open(item))    # get image
        input_var = torch.autograd.Variable(test_image, volatile=True).unsqueeze(0).cuda()
        test_image_feature=model(input_var) # get feature
        test_result = []
        for pig in class_index:
            train_data = np.load(os.path.join(train_dir, str(pig)+'_resnet50.npy'))
            num = 0
            for train_sqe in range(int(train_data.shape[0])):
                euclidean_distance = pairwise_distance1(test_image_feature[:,:,0,0].data.cpu().numpy(), train_data[train_sqe])
                num += euclidean_distance
            average = num/int((train_data.shape[0]))
            test_result.append(1/average**2)
        output = nn.functional.softmax(torch.from_numpy(np.array(test_result)))
        f_val.writelines(test_image_name+','+str(1)+','+str('%.8f'%output.data[0])+'\n')
        f_val.writelines(test_image_name+','+str(2)+','+str('%.8f'%output.data[11])+'\n')
        f_val.writelines(test_image_name+','+str(3)+','+str('%.8f'%output.data[22])+'\n')
        f_val.writelines(test_image_name+','+str(4)+','+str('%.8f'%output.data[24])+'\n')
        f_val.writelines(test_image_name+','+str(5)+','+str('%.8f'%output.data[25])+'\n')
        f_val.writelines(test_image_name+','+str(6)+','+str('%.8f'%output.data[26])+'\n')
        f_val.writelines(test_image_name+','+str(7)+','+str('%.8f'%output.data[27])+'\n')
        f_val.writelines(test_image_name+','+str(8)+','+str('%.8f'%output.data[28])+'\n')
        f_val.writelines(test_image_name+','+str(9)+','+str('%.8f'%output.data[29])+'\n')
        f_val.writelines(test_image_name+','+str(10)+','+str('%.8f'%output.data[1])+'\n')
        f_val.writelines(test_image_name+','+str(11)+','+str('%.8f'%output.data[2])+'\n')
        f_val.writelines(test_image_name+','+str(12)+','+str('%.8f'%output.data[3])+'\n')
        f_val.writelines(test_image_name+','+str(13)+','+str('%.8f'%output.data[4])+'\n')
        f_val.writelines(test_image_name+','+str(14)+','+str('%.8f'%output.data[5])+'\n')
        f_val.writelines(test_image_name+','+str(15)+','+str('%.8f'%output.data[6])+'\n')
        f_val.writelines(test_image_name+','+str(16)+','+str('%.8f'%output.data[7])+'\n')
        f_val.writelines(test_image_name+','+str(17)+','+str('%.8f'%output.data[8])+'\n')
        f_val.writelines(test_image_name+','+str(18)+','+str('%.8f'%output.data[9])+'\n')
        f_val.writelines(test_image_name+','+str(19)+','+str('%.8f'%output.data[10])+'\n')
        f_val.writelines(test_image_name+','+str(20)+','+str('%.8f'%output.data[12])+'\n')
        f_val.writelines(test_image_name+','+str(21)+','+str('%.8f'%output.data[13])+'\n')
        f_val.writelines(test_image_name+','+str(22)+','+str('%.8f'%output.data[14])+'\n')
        f_val.writelines(test_image_name+','+str(23)+','+str('%.8f'%output.data[15])+'\n')
        f_val.writelines(test_image_name+','+str(24)+','+str('%.8f'%output.data[16])+'\n')
        f_val.writelines(test_image_name+','+str(25)+','+str('%.8f'%output.data[17])+'\n')
        f_val.writelines(test_image_name+','+str(26)+','+str('%.8f'%output.data[18])+'\n')
        f_val.writelines(test_image_name+','+str(27)+','+str('%.8f'%output.data[19])+'\n')
        f_val.writelines(test_image_name+','+str(28)+','+str('%.8f'%output.data[20])+'\n')
        f_val.writelines(test_image_name+','+str(29)+','+str('%.8f'%output.data[21])+'\n')
        f_val.writelines(test_image_name+','+str(30)+','+str('%.8f'%output.data[23])+'\n')
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