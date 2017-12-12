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
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
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
best_loss1 = 1000


def main():
    global args, best_prec1,best_loss1
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

    # for parma in model.parameters():  # if we only need train the fc, we need thses lines
    #    parma.requires_grad = False

    if args.arch.startswith('vgg'):
        # model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
        #                                torch.nn.ReLU(),
        #                                torch.nn.Dropout(p=0.5),
        #                                torch.nn.Linear(4096, 4096),
        #                                torch.nn.ReLU(),
        #                                torch.nn.Dropout(p=0.5),
        #                                torch.nn.Linear(4096, 30))

        mod = list(model.classifier.children())
        mod.pop()
        mod.append(torch.nn.Linear(4096, 30))
        new_classifier = torch.nn.Sequential(*mod)
        model.classifier = new_classifier
        
        
    elif args.arch.startswith('alexnet'):
        mod = list(model.classifier.children())
        mod.pop()
        mod.append(torch.nn.Linear(4096, 30))
        new_classifier = torch.nn.Sequen(*mod)
        model.classifier = new_classifier
    elif args.arch.startswith('densenet'):
        model.classifier=torch.nn.Linear(1920, 30)    
    else:
        model.fc=torch.nn.Linear(2048, 30) 
        # mod = list(model.children())
        # #mod.pop()
        # #mod.append(nn.Dropout())
        # #mod.append(torch.nn.Linear(2048, 30))
        # new_classifier = torch.nn.Sequential(*mod)
        # model = new_classifier
    
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
            self.dropout=nn.Dropout()
            self.fc = model.fc

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
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.fc(x)

            return x
    my_model=MyResNet(model)
    model=my_model.cuda()
    print(model)


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # for index, parma in enumerate(model.classifier.parameters()):
    #     if index == 6:
    #         parma.requires_grad = True
#####################
    # optimizer = torch.optim.SGD(model.fc.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.fc.parameters(), args.lr,  # if we just train fc, we need these lines
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), args.lr,   # if we train all the model, we need these lines
                                weight_decay=args.weight_decay)
#######################
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
######################
    logpath_train='./allset/'+args.arch+'/'
    txtname='train.csv'
    if not os.path.exists(logpath_train):
        os.makedirs(logpath_train)
    if os.path.exists(logpath_train+txtname):
        os.remove(logpath_train+txtname)
    global f_train
    f_train=file(logpath_train+txtname,'a+')

#################
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            #transforms.CenterCrop([224.224]),
            transforms.Scale([224,224]),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    print(traindir)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)
        print(optimizer.param_groups[0]['lr'])
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # remember best prec@1 and save checkpoint
        is_best = True
        save_checkpoint({   # this is a function
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, epoch ,is_best)
       ######
        # save_path=logpath_val+args.data.split('/')[-1]
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % (save_path, epoch))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            f_train.writelines('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} {loss.avg:.4f}\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f}) \n'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

def save_checkpoint(state, epoch,is_best, filename='_checkpoint.pth.tar'):
    global args
    if not os.path.exists('allset'):
        os.makedirs('allset')
    tmp_name = str(epoch+1) + '_' +filename
    torch.save(state, 'allset/'+tmp_name)
    if is_best:
        shutil.copyfile('allset/'+tmp_name, 'allset/model_best.pth.tar') # tmp_name includes sequence


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
    lr = args.lr * (0.1 ** (epoch // 40))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()