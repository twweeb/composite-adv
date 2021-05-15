from __future__ import print_function
import os
import argparse
import shutil
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
import numpy as np
import composite_pgd, sinkhorn_ops
from models.wideresnet import *
from models.resnet import *
from trades import trades_loss
from math import pi

torch.distributed.init_process_group(backend="nccl")
torch.autograd.set_detect_anomaly(True)


def list_type(s):
    try:
        return tuple(sorted(map(int, s.split(','))))
    except:
        raise argparse.ArgumentTypeError("List must be (x,x,....,x) ")


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--model_path', default='./adv_train', help='directory of model for loading checkpoint')
parser.add_argument("--enable", type=list_type, default=(0, 1), help="list of enabled attacks")
parser.add_argument("--log_filename", default='logfile.csv', help="filename of output log")

args = parser.parse_args()

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
# device = torch.device("cuda" if use_cuda else "cpu")
# local_rank = torch.distributed.get_rank()
print('local rank --> {}'.format(args.local_rank))
torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank)
args.nprocs = torch.cuda.device_count()
kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
indices = list(np.random.randint(0, len(trainset), int(len(trainset))))
train_sampler = SubsetRandomSampler(indices)

print(len(train_sampler))
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, **kwargs,
                                           sampler=DistributedSampler(train_sampler))
print(len(train_loader))
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

# start_num = [0,1,5,10,15] # 0: No Attack
# iter_num = [5,10,15,20,25] # 0: No Attack
start_num = 1
iter_num = 1
hue_epsilon = 0.5
sat_epsilon = 0.5
rot_epsilon = 30
bright_epsilon = 0.5
contrast_epsilon = 0.5
linf_epsilon = 1 / 1275
step_size = 0.05
linf_step_size = 1 / 2550
multiple_rand_start = True
flag_target = False
debug = False
together = True
# linf_epsilons_space = [1/2550,2/2550,5/2550,1/255,2/255]
# hue_epsilons_space = [0.1, 0.2, 0.3, 0.4, 0.5]
# rot_epsilons_space = [5.,10.,15.,20.,25.,30.]
sequence_single = [(0,), (1,), (2,), (3,), (4,), (5,)]
attack_name = ["Hue", "Saturate", "Rotate", "Bright", "Contrast", "L-Infinity"]


def train(args, model, device, train_loader, optimizer, epoch, local_rank):
    model.train()
    pgd_attack = composite_pgd.PGD(model, device, args.enable)
    pgd_attack.parse_params(hue_epsilon=(-pi, pi), sat_epsilon=(0.7, 1.3), rot_epsilon=(-15, 15),
                            bright_epsilon=(-0.3, 0.3), contrast_epsilon=(0.7, 1.3), linf_epsilon=(-8 / 255, 8 / 255),
                            start_num=start_num, iter_num=iter_num, multiple_rand_start=True, order_schedule='random')

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        # loss = F.cross_entropy(model(data), target)

        loss = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           local_rank=local_rank,
                           optimizer=optimizer,
                           pgd_attack=pgd_attack,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta)

        torch.distributed.barrier()
        reduced_loss = reduce_mean(loss, args.nprocs)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Process Id: {} Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}\t Avg. Loss: {:.6f}'.format(
                dist.get_rank(), epoch, batch_idx * len(data), len(train_loader) * args.batch_size,
                                        100. * batch_idx / len(train_loader), loss.item(), reduced_loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    if dist.get_rank() == 0:
        print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            train_loss, correct, len(train_loader) * args.batch_size,
                                 100. * correct / (len(train_loader) * args.batch_size)))
    training_accuracy = 100. * correct / (len(train_loader) * args.batch_size)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    if dist.get_rank() == 0:
        print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # init model, ResNet18() can be also used here for training

    model = WideResNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    epoch = 0
    best_acc1 = .0
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print('load model for retraining ', args.model_path)
        print('epoch --> ', epoch)

    if torch.cuda.device_count() > 1 and args.local_rank != -1:
        print("Init Process Id: {} | Total GPU: {}".format(args.local_rank, torch.cuda.device_count()))
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False,
                    find_unused_parameters=True)

    for e in range(epoch, epoch + args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch + e + 1)

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch + e + 1, args.local_rank)

        # evaluation on natural examples

        # if dist.get_rank() == 0:
        train_loss, train_acc1 = eval_train(model, device, train_loader)
        test_loss, test_acc1 = eval_test(model, device, test_loader)

        # remember best acc@1 and save checkpoint
        is_best = test_acc1 > best_acc1
        best_acc1 = max(test_acc1, best_acc1)

        # save checkpoint
        if dist.get_rank() == 0:

            print("Best Test Accuracy: {}%".format(best_acc1))

            filename = os.path.join(model_dir, 'model-epoch{}.pt'.format(epoch + e + 1))
            torch.save({
                'epoch': epoch + e + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc1': best_acc1
            }, filename)
            print('Save model: {}'.format(os.path.join(model_dir, 'model-epoch{}.pt'.format(epoch + e + 1))))
            if is_best:
                print("Copy best model!")
                shutil.copyfile(filename, os.path.join(model_dir, 'model_best.pth'))
            print('================================================================')
            with open(args.log_filename, 'a+') as f:
                csv_write = csv.writer(f)
                # data_row = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch_start)), epoch_end - epoch_start]
                data_row = [epoch + e + 1, train_loss, train_acc1, test_loss, test_acc1, best_acc1]
                csv_write.writerow(data_row)
            # torch.save(model.state_dict(),
            #            os.path.join(model_dir, 'model-wideres-epoch{}.pt'.format(epoch)))
            # torch.save(optimizer.state_dict(),
            #            os.path.join(model_dir, 'opt-wideres-checkpoint_epoch{}.tar'.format(epoch)))


if __name__ == '__main__':
    main()
