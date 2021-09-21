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
import random
import composite_pgd, sinkhorn_ops
from models.wideresnet import *
# from models.resnet import *
from robustness.cifar_models.resnet import *
from trades import trades_loss
from math import pi

# torch.distributed.init_process_group(backend="nccl")
# torch.autograd.set_detect_anomaly(True)


def list_type(s):
    try:
        return tuple(map(int, s.split(',')))
    except:
        raise argparse.ArgumentTypeError("List must be (x,x,....,x) ")

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
parser.add_argument('--beta', default=6.0, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--arch', default='wideresnet',
                    help='architecture of model')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--dist', default='comp', type=str,
                    help='distance metric')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--order', default='random', type=str, help='specify the order')
parser.add_argument('--model_path', default='./adv_train', help='directory of model for loading checkpoint')
parser.add_argument("--enable", type=list_type, default=(0, 1), help="list of enabled attacks")
parser.add_argument("--power", type=str, default='strong', help="level of attack power")
parser.add_argument("--linf_loss", type=str, default='ce', help="loss for linf-attack, ce or kl")
parser.add_argument("--log_filename", default='logfile.csv', help="filename of output log")

args = parser.parse_args()

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
# local_rank = torch.distributed.get_rank()
# print('local rank --> {}'.format(args.local_rank))
# torch.cuda.set_device(args.local_rank)
# args.nprocs = torch.cuda.device_count()
kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

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
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, **kwargs)
                                           #sampler=DistributedSampler(train_sampler))
print(len(train_loader))
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

# start_num = [0,1,5,10,15] # 0: No Attack
# iter_num = [5,10,15,20,25] # 0: No Attack
start_num = 1
iter_num = 1
inner_iter_num = 0

# linf_epsilons_space = [1/2550,2/2550,5/2550,1/255,2/255]
# hue_epsilons_space = [0.1, 0.2, 0.3, 0.4, 0.5]
# rot_epsilons_space = [5.,10.,15.,20.,25.,30.]
sequence_single = [(0,), (1,), (2,), (3,), (4,), (5,)]
attack_name = ["Hue", "Saturate", "Rotate", "Bright", "Contrast", "L-Infinity"]


def train(args, model, train_loader, optimizer, epoch, local_rank, pgd_attack):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        # calculate robust loss
        # loss = F.cross_entropy(model(data), target)

        loss, lost_nat, loss_adv = trades_loss(model=model,
                                               x_natural=data,
                                               y=target,
                                               local_rank=local_rank,
                                               optimizer=optimizer,
                                               pgd_attack=pgd_attack,
                                               step_size=args.step_size,
                                               epsilon=args.epsilon,
                                               perturb_steps=args.num_steps,
                                               beta=args.beta,
                                               distance=args.dist)

        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Robust Loss: {:.6f}, Natural Loss: {:.6f}, Adv Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * args.batch_size,
                100. * batch_idx / len(train_loader), loss.item(), lost_nat.item(), loss_adv.item()))


def eval_train(model, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader) * args.batch_size,
        100. * correct / (len(train_loader) * args.batch_size)))
    training_accuracy = 100. * correct / (len(train_loader) * args.batch_size)
    return train_loss, training_accuracy


def eval_test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
           test_loss, correct, len(test_loader.dataset),
           100. * correct / len(test_loader.dataset)))
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch, no_improve):
    """decrease the learning rate"""
    lr = args.lr
    if no_improve >= 10 and epoch < 99:
        lr = args.lr * 0.1 ** (no_improve // 10)
    elif epoch >= 75:
        lr = args.lr * 0.1
    elif epoch >= 90:
        lr = args.lr * 0.01
    elif epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # init model, ResNet18() can be also used here for training

    if args.arch == 'wideresnet':
        model = WideResNet().cuda()
    elif args.arch == 'resnet50':
        model = ResNet50().cuda()
    else:
        print('Model architecture not specified.')

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    epoch = 0
    best_acc1 = .0
    no_improve = 0
    if os.path.exists(args.model_path):
        if 'pretrained_model' in args.model_path:
            from robustness.model_utils import make_and_restore_model
            from robustness.datasets import DATASETS
            _dataset = DATASETS['cifar']('../data')
            model, checkpoint = make_and_restore_model(arch=args.arch,
                                                       dataset=_dataset, resume_path=args.model_path)
            print('model successfully loaded.')
            filename = os.path.join(model_dir, 'model-epoch0.pt')
            torch.save({
                'epoch': 0,
                'model_state_dict': model.model.state_dict(),
                'best_acc1': 0.0,
                'no_improve': 0,
            }, filename)
        else:
            try:
                checkpoint = torch.load(args.model_path) #, pickle_module=dill)
                if isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        model.load_state_dict(checkpoint['model'])
                        print('model')
                    elif 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])

                    if 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if 'epoch' in checkpoint:
                        epoch = checkpoint['epoch'] + 1
                    if 'best_acc1' in checkpoint:
                        best_acc1 = checkpoint['best_acc1']
                    if 'no_improve' in checkpoint:
                        no_improve = checkpoint['no_improve']

                    print("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
                    print('best_accuracy --> ', best_acc1)
                    print('No improve --> ', no_improve)

            except RuntimeError as error:
                raise error  # type: ignore

    pgd_attack = composite_pgd.PGD(model, args.enable, attack_power=args.power, linf_loss=args.linf_loss,
                                   start_num=start_num, iter_num=iter_num, inner_iter_num=inner_iter_num,
                                   multiple_rand_start=True, order_schedule=args.order)

    # test_loss, test_acc1 = eval_test(model, test_loader)
    # print("Test Accuracy: {}%".format(test_acc1))
    for e in range(epoch, epoch + args.epochs):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, e, no_improve)

        # adversarial training
        train(args, model, train_loader, optimizer, e, args.local_rank, pgd_attack)

        # evaluation on natural examples
        train_loss, train_acc1 = eval_train(model, train_loader)
        test_loss, test_acc1 = eval_test(model, test_loader)

        # remember best acc@1 and save checkpoint
        is_best = test_acc1 > best_acc1
        best_acc1 = max(test_acc1, best_acc1)
        if is_best:
            no_improve = no_improve - (no_improve % 10)
        else:
            no_improve = no_improve + 1
        print("No improve: {}".format(no_improve))

        # save checkpoint
        print("Best Test Accuracy: {}%".format(best_acc1))

        filename = os.path.join(model_dir, 'model-epoch{}.pt'.format(e))
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc1': best_acc1,
            'no_improve': no_improve,
        }, filename)
        print('Save model: {}'.format(os.path.join(model_dir, 'model-epoch{}.pt'.format(e))))
        if is_best:
            print("Copy best model!")
            shutil.copyfile(filename, os.path.join(model_dir, 'model_best.pth'))
        print('================================================================')
        with open(args.log_filename, 'a+') as f:
            csv_write = csv.writer(f)
            # data_row = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch_start)), epoch_end - epoch_start]
            data_row = [e, train_loss, train_acc1, test_loss, test_acc1, best_acc1]
            csv_write.writerow(data_row)
            # torch.save(model.state_dict(),
            #            os.path.join(model_dir, 'model-wideres-epoch{}.pt'.format(epoch)))
            # torch.save(optimizer.state_dict(),
            #            os.path.join(model_dir, 'opt-wideres-checkpoint_epoch{}.tar'.format(epoch)))


if __name__ == '__main__':
    main()
