from __future__ import print_function
import os
import warnings
import argparse
import shutil
import csv
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import datasets, transforms
import numpy as np
import random
from models.wideresnet import *
from torchvision.models import resnet50
import matplotlib.pyplot as plt
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
import sys
from trades import trades_loss


def list_type(s):
    try:
        return tuple(map(int, s.split(',')))
    except:
        raise argparse.ArgumentTypeError("List must be (x,x,....,x) ")


parser = argparse.ArgumentParser(description='PyTorch Tiny ImageNet Natural Training')
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
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--arch', default='wideresnet',
                    help='architecture of model')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--mode', default='natural', type=str,
                    help='specify training mode (natural or adv_train)')
parser.add_argument('--order', default='random', type=str, help='specify the order')
parser.add_argument('--model_path', default='./adv_train', help='directory of model for loading checkpoint')
parser.add_argument("--enable", type=list_type, default=(0, 1), help="list of enabled attacks")
parser.add_argument("--log_filename", default='logfile.csv', help="filename of output log")

parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0

def main():
    # settings
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    model, epoch, optimizer, criterion, best_acc1 = load_model(args, ngpus_per_node)
    cudnn.benchmark = True

    # setup data loader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # normalize,
    ])

    DATA_DIR = '/work/hsiung1024/imagenet'  # Original images come in shapes of [3,64,64]
    # Define training and validation data paths
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VALID_DIR = os.path.join(DATA_DIR, 'val')

    train_loader = generate_dataloader(TRAIN_DIR, "train", transform_train, workers=ngpus_per_node*4,
                                       batch_size=args.batch_size, distributed=args.distributed)
    test_loader = generate_dataloader(VALID_DIR, "val", transform_test, workers=ngpus_per_node*4,
                                      batch_size=args.batch_size)
    train(model, epoch, optimizer, criterion, best_acc1, train_loader, test_loader, args)


def generate_dataloader(data, name, transform, workers, batch_size, distributed=False):
    if data is None:
        return None

    if transform is None:
        dataset = datasets.ImageFolder(data, transform=transforms.ToTensor())
    else:
        dataset = datasets.ImageFolder(data, transform=transform)

    if name == "train":
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            train_sampler = None
    else:
        train_sampler = None

    # Wrap image dataset (defined above) in dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=(train_sampler is None),
                            num_workers=workers,
                            pin_memory=True,
                            sampler=train_sampler)

    return dataloader


def load_model(args, ngpus_per_node):
    # init model, ResNet18() can also be used for training here

    if args.arch == 'wideresnet':
        model = WideResNet(num_classes=200).cuda()
    elif args.arch == 'resnet50':
        model = resnet50(pretrained=True)
    else:
        print('Model architecture not specified.')

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    epoch = 0
    best_acc1 = .0
    if os.path.exists(args.model_path):
        if 'pretrained_model' in args.model_path:
            from robustness.model_utils import make_and_restore_model
            from robustness.datasets import DATASETS
            _dataset = DATASETS['cifar']('../data')
            model, checkpoint = make_and_restore_model(arch=args.arch,
                                                       dataset=_dataset, resume_path=args.model_path)
            print('model successfully loaded.')
            filename = os.path.join(args.model_dir, 'model-epoch0.pt')
            torch.save({
                'epoch': 0,
                'model_state_dict': model.model.state_dict(),
                'best_acc1': 0.0,
            }, filename)
        else:
            try:
                checkpoint = torch.load(args.model_path)  # , pickle_module=dill)
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

                    print("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
                    print('best_accuracy --> ', best_acc1)

            except RuntimeError as error:
                raise error  # type: ignore
    return model, epoch, optimizer, criterion, best_acc1


def eval_train(model, train_loader, args):
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


def visualize_dataset(viz_dataset, viz_dataloader, _model=None):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    data_features, data_labels = next(iter(viz_dataloader))
    if _model is not None:
        _model.eval()
        data_features_cuda = data_features.cuda()
        output = _model(data_features_cuda)
        data_labels = output.max(1, keepdim=True)[1].cpu()

    print(f"Feature batch shape: {data_features.size()}")
    print(f"Labels batch shape: {data_labels.size()}")
    # print(f"Labels: {data_labels}")

    for i in range(1, cols * rows + 1):
        img, label = data_features[i], data_labels[i]
        figure.add_subplot(rows, cols, i)
        plt.title(",".join(viz_dataset.nid_to_words[viz_dataset.ids[label]]))
        plt.axis("off")
        npimg = np.transpose(img.squeeze().numpy(), (1, 2, 0))
        plt.imshow(npimg)

    if _model is not None:
        plt.savefig('images/tiny_imagenet (model).pdf')
    else:
        plt.savefig('images/tiny_imagenet (ground_truth).pdf')
    plt.show()


def train_ep(args, model, train_loader, optimizer, criterion, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        # zero gradient
        optimizer.zero_grad()
        # calculate robust loss
        logits = model(data)
        loss = criterion(logits, target)

        # loss, lost_nat, loss_adv = trades_loss(model=model,
        #                                        x_natural=data,
        #                                        y=target,
        #                                        local_rank=local_rank,
        #                                        optimizer=optimizer,
        #                                        pgd_attack=pgd_attack,
        #                                        step_size=args.step_size,
        #                                        epsilon=args.epsilon,
        #                                        perturb_steps=args.num_steps,
        #                                        beta=args.beta,
        #                                        distance=args.dist)

        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader) * args.batch_size,
                           100. * batch_idx / len(train_loader), loss.item()))


def train(model, epoch, optimizer, criterion, best_acc1, train_loader, test_loader, args):
    test_loss, test_acc1 = eval_test(model, test_loader)
    print("Test Accuracy: {}%".format(test_acc1))
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    liveloss = PlotLosses(outputs=[MatplotlibPlot(figpath="./result/imagenet_train_loss.png")])

    for e in range(epoch, epoch + args.epochs):

        # adversarial training
        train_ep(args, model, train_loader, optimizer, criterion, e)

        exp_lr_scheduler.step()
        # evaluation on natural examples
        train_loss, train_acc1 = eval_train(model, train_loader, args)
        test_loss, test_acc1 = eval_test(model, test_loader)

        # remember best acc@1 and save checkpoint
        is_best = test_acc1 > best_acc1
        best_acc1 = max(test_acc1, best_acc1)

        # save checkpoint
        print("Best Test Accuracy: {}%".format(best_acc1))

        filename = os.path.join(args.model_dir, 'model-epoch{}.pt'.format(e))
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc1': best_acc1,
        }, filename)
        # print('Save model: {}'.format(os.path.join(model_dir, 'model-epoch{}.pt'.format(e))))
        if is_best:
            print("Save best model (epoch {})!".format(e))
            # shutil.copyfile(filename, os.path.join(model_dir, 'model_best.pth'))
            print('Save model: {}'.format(os.path.join(args.model_dir, 'model_best.pth')))
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
        liveloss.update({
            'log loss': train_loss,
            'val_log loss': test_loss,
            'accuracy': train_acc1,
            'val_accuracy': test_acc1
        })

        liveloss.send()


if __name__ == '__main__':
    main()
    # visualize_dataset(testset, test_loader, _model=model)
