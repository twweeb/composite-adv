from __future__ import print_function
import warnings
import argparse
import shutil
import csv
import dill
import builtins
import time
import os
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import datasets, transforms
import numpy as np
import random
from torchvision.models import resnet50
import matplotlib.pyplot as plt
from generalized_order_attack.utilities import imshow, InputNormalize
from torch.autograd import Variable
from generalized_order_attack.attacks import *


def list_type(s):
    try:
        return tuple(map(int, s.split(',')))
    except:
        raise argparse.ArgumentTypeError("List must be (x,x,....,x) ")


parser = argparse.ArgumentParser(description='PyTorch Tiny ImageNet Natural Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=1e-4,
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
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--print-freq', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--arch', default='wideresnet',
                    help='architecture of model')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--dist', default='comp', type=str,
                    help='distance metric')
parser.add_argument('--mode', default='natural', type=str,
                    help='specify training mode (natural or adv_train)')
parser.add_argument('--debug', action='store_true',
                    help='Train Only One Epoch and print training images.')
parser.add_argument('--checkpoint', type=str, default=None, help='path of checkpoint')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--order', default='random', type=str, help='specify the order')
parser.add_argument('--stat-dict', type=str, default=None,
                    help='key of stat dict in checkpoint')
parser.add_argument("--enable", type=list_type, default=(0, 1), help="list of enabled attacks")
parser.add_argument("--power", type=str, default='strong', help="level of attack power")
parser.add_argument("--log_filename", default='logfile.csv', help="filename of output log")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--dist-file', default=None, type=str, help='distributed config')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


epoch = 0
best_acc1 = .0
iteration = 0
train_loader_len = None

start_num = 1
iter_num = 1
inner_iter_num = 7

sequence_single = [(0,), (1,), (2,), (3,), (4,), (5,)]
attack_name = ["Hue", "Saturate", "Rotate", "Bright", "Contrast", "L-Infinity"]


def find_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.


def main():
    # settings
    args = parser.parse_args()

    try:
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
    except FileExistsError:
        pass

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

    # slurm available
    if args.world_size == -1 and "SLURM_NPROCS" in os.environ:
        args.world_size = int(os.environ["SLURM_NPROCS"])
        args.rank = int(os.environ["SLURM_PROCID"])
        jobid = os.environ["SLURM_JOBID"]
        hostfile = "dist_url." + jobid  + ".txt"
        if args.dist_file is not None:
            args.dist_url = "file://{}.{}".format(os.path.realpath(args.dist_file), jobid)
        elif args.rank == 0:
            import socket
            ip = socket.gethostbyname(socket.gethostname())
            port = find_free_port()
            args.dist_url = "tcp://{}:{}".format(ip, port)
            with open(hostfile, "w") as f:
                f.write(args.dist_url)
        else:
            while not os.path.exists(hostfile):
                time.sleep(1)
            with open(hostfile, "r") as f:
                args.dist_url = f.read()
        print("dist-url:{} at PROCID {} / {}".format(args.dist_url, args.rank, args.world_size))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1, epoch, iteration
    args.gpu = gpu

    if args.rank != -1:
        if args.gpu is not None and args.gpu % ngpus_per_node == 0:
            print("Node {} uses {} GPUs for training, ".format(args.rank, ngpus_per_node) +
                  "multi-processing." if args.multiprocessing_distributed else "single-processing, multi-threading.")
        elif args.gpu is None:
            print("Node {} uses {} GPUs for training, ".format(args.rank, ngpus_per_node) +
                  "multi-processing." if args.multiprocessing_distributed else "single-processing, multi-threading.")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.multiprocessing_distributed:
        def print_pass(*args, sep=' ', end='\n', file=None):
            pass
        if args.rank > 0:
            builtins.print = print_pass
        elif args.gpu is not None and args.gpu % ngpus_per_node != 0:
            builtins.print = print_pass

    model = load_model(args)

    # Send to GPU
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            if args.arch == 'wideresnet':
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                                  find_unused_parameters=True)
            else:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
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

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.checkpoint:
        if args.stat_dict == 'madry':
            pass
        elif args.stat_dict == 'gat':
            checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))  # , pickle_module=dill)
            assert isinstance(checkpoint, dict)
            sd = checkpoint['model_state_dict']
            # sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                  weight_decay=args.weight_decay)
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'epoch' in checkpoint:
                epoch = checkpoint['epoch'] + 1
            if 'best_acc1' in checkpoint:
                best_acc1 = checkpoint['best_acc1']

            print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, epoch))
            print('best_accuracy --> ', best_acc1)
        else:
            pass

    cudnn.benchmark = True

    # setup data loader
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    DATA_DIR = '/work/hsiung1024/imagenet/'  # Original images come in shapes of [3,64,64]
    # Define training and validation data paths
    TRAIN_LMDB = os.path.join(DATA_DIR, 'train')
    VALID_LMDB = os.path.join(DATA_DIR, 'val')

    train_loader, train_sampler = generate_dataloader(TRAIN_LMDB, "train", transform_train, args)
    test_loader, _ = generate_dataloader(VALID_LMDB, "val", transform_test, args)

    if args.evaluate:
        validate(test_loader, model, criterion, args)
        return
    train(model, optimizer, criterion, train_loader, train_sampler, test_loader, args, ngpus_per_node)


def generate_dataloader(data, name, transform, args):
    if data is None:
        return None

    if transform is None:
        dataset = datasets.ImageFolder(data, transform=transforms.ToTensor())
    else:
        dataset = datasets.ImageFolder(data, transform=transform)

    if name == "train":
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            train_sampler = None
    else:
        train_sampler = None

    # Wrap image dataset (defined above) in dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=(train_sampler is None),
                            num_workers=args.workers,
                            pin_memory=True,
                            sampler=train_sampler)

    return dataloader, train_sampler


def load_model(args):
    # init model, ResNet18() can also be used for training here

    if args.arch == 'wideresnet':
        raise NotImplementedError()
    elif args.arch == 'resnet50':
        if args.checkpoint is not None:
            if args.stat_dict == 'madry':
                from robustness.datasets import DATASETS
                from robustness.attacker import AttackerModel
                _dataset = DATASETS['imagenet']('/work/hsiung1024/imagenet')
                model = _dataset.get_model(args.arch, False)
                model = AttackerModel(model, _dataset)

                if args.checkpoint and os.path.isfile(args.checkpoint):
                    print("=> loading checkpoint '{}'".format(args.checkpoint))
                    checkpoint = torch.load(args.checkpoint, pickle_module=dill, map_location=torch.device('cpu'))

                    # Makes us able to load models saved with legacy versions
                    state_dict_path = 'model'
                    if not ('model' in checkpoint):
                        state_dict_path = 'state_dict'

                    sd = checkpoint[state_dict_path]
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                    model.load_state_dict(sd)
                    print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, checkpoint['epoch']))
                elif args.checkpoint:
                    error_msg = "=> no checkpoint found at '{}'".format(args.checkpoint)
                    raise ValueError(error_msg)

                model = nn.Sequential(
                    InputNormalize(torch.tensor([0.485, 0.456, 0.406]),
                                   torch.tensor([0.229, 0.224, 0.225])),
                    model.model
                )
            elif args.stat_dict == 'gat':
                from robustness.datasets import DATASETS
                from robustness.attacker import AttackerModel
                _dataset = DATASETS['imagenet']('/work/hsiung1024/imagenet')
                model = _dataset.get_model(args.arch, False)
                model = AttackerModel(model, _dataset)
                model = nn.Sequential(
                    InputNormalize(torch.tensor([0.485, 0.456, 0.406]),
                                   torch.tensor([0.229, 0.224, 0.225])),
                    model.model
                )
            else:
                raise NotImplementedError()
        else:
            if args.stat_dict == 'from_scratch':
                model = resnet50(pretrained=False)
                model = nn.Sequential(
                    InputNormalize(torch.tensor([0.485, 0.456, 0.406]),
                                   torch.tensor([0.229, 0.224, 0.225])),
                    model
                )
            elif args.stat_dict == 'finetune_natural':
                model = resnet50(pretrained=True)
                model = nn.Sequential(
                    InputNormalize(torch.tensor([0.485, 0.456, 0.406]),
                                   torch.tensor([0.229, 0.224, 0.225])),
                    model
                )
            else:
                raise ValueError()
    else:
        print('Model architecture not specified.')
        raise ValueError()

    return model


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


def train_ep(args, model, train_loader, pgd_attack, optimizer, criterion):
    global epoch, iteration
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    model.train()

    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, args)
        iteration += 1

        if args.gpu is not None:
            data = data.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # clean training
        if args.mode == 'natural':
            logits = model(data)
            loss = criterion(logits, target)

        # adv training normal
        elif args.mode == 'adv_train_madry':
            model.eval()
            # generate adversarial example
            if args.gpu is not None:
                data_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda(args.gpu, non_blocking=True).detach()
            else:
                data_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach()

            data_adv = pgd_attack(data_adv, target)
            data_adv = Variable(torch.clamp(data_adv, 0.0, 1.0), requires_grad=False)
            if args.debug:
                imshow(data, model, classes_map,
                       ground_truth=target, save_file='images/train/imagenet_adv_train_madry (clean).pdf', show=False)
                imshow(data_adv, model, classes_map,
                       ground_truth=target, save_file='images/train/imagenet_adv_train_madry (attack).pdf', show=False)
                break

            model.train()

            logits = model(data_adv)
            loss = criterion(logits, target)

        # adv training by trades
        elif args.mode == 'adv_train_trades':
            # TRADE Loss would require more memory.

            model.eval()
            batch_size = len(data)
            # generate adversarial example
            if args.gpu is not None:
                data_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda(args.gpu, non_blocking=True).detach()
            else:
                data_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach()

            data_adv = pgd_attack(data_adv, target)
            data_adv = Variable(torch.clamp(data_adv, 0.0, 1.0), requires_grad=False)
            if args.debug:
                imshow(data, model, classes_map,
                       ground_truth=target, save_file='images/train/imagenet_adv_train_trades (clean).pdf', show=False)
                imshow(data_adv, model, classes_map,
                       ground_truth=target, save_file='images/train/imagenet_adv_train_trades (attack).pdf', show=False)
                break

            model.train()

            # calculate robust loss
            logits = model(data)
            loss_natural = F.cross_entropy(logits, target)
            loss_robust = (1.0 / batch_size) * F.kl_div(F.log_softmax(model(data_adv), dim=1),
                                                        F.softmax(model(data), dim=1))
            loss = loss_natural + args.beta * loss_robust

        else:
            print("Not Specify Training Mode.")
            raise ValueError()

        # measure adv_exp generating time
        data_time.update(time.time() - end)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0].item(), data.size(0))
        top5.update(acc5[0].item(), data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            progress.display(batch_idx)

    return top1.avg, top5.avg, losses.avg


def train(model, optimizer, criterion, train_loader, train_sampler, test_loader, args, ngpus_per_node):
    global best_acc1, epoch, iteration, train_loader_len
    train_loader_len = len(train_loader)
    iteration = epoch*train_loader_len
    pgd_attack = CompositeAttack(model, args.enable, mode='train', attack_power=args.power,
                                 start_num=start_num, iter_num=iter_num, inner_iter_num=inner_iter_num,
                                 multiple_rand_start=True, order_schedule=args.order)

    for e in range(epoch, epoch + args.epochs):
        epoch = e
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adversarial training
        train_acc1, train_acc5, train_loss = train_ep(args, model, train_loader, pgd_attack, optimizer, criterion)

        test_acc1, test_acc5, test_loss = validate(test_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = test_acc1 > best_acc1
        best_acc1 = max(test_acc1, best_acc1)

        if not args.multiprocessing_distributed or \
                (args.multiprocessing_distributed and args.rank <= 0 and args.gpu % ngpus_per_node == 0):
            # if args.gpu is None or args.gpu == 0:
            print("Best Test Accuracy: {}%".format(best_acc1))
            # save checkpoint
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'model_state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.model_dir)
            with open(args.log_filename, 'a+') as f:
                csv_write = csv.writer(f)
                data_row = [epoch,
                            train_loss, train_acc1, train_acc5,
                            test_loss, test_acc1, test_acc5,
                            best_acc1]
                csv_write.writerow(data_row)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, model_dir=None):
    filename = os.path.join(model_dir, 'model-epoch{}.pt'.format(state['epoch']))
    torch.save(state, filename)
    print('Save model: {}'.format(filename))
    if is_best:
        best_cp = os.path.join(model_dir, 'model_best.pth')
        print("Save best model (epoch {})!".format(state['epoch']))
        shutil.copyfile(filename, best_cp)
        print('Save model: {}'.format(best_cp))
    print('================================================================')


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def forward(self, input_tensor):
        return self.normalize(input_tensor)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, e, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    global train_loader_len
    lr = args.lr * (0.1 ** (e // 30))
    # ramp-up learning rate for SGD
    if e < 5 and args.lr >= 0.1:
        lr = (iteration + 1) / (5 * train_loader_len) * args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
    # visualize_dataset(testset, test_loader, _model=model)
