from typing import Dict, List
import torch
import csv
import argparse
import os
from generalized_order_attack.attacks import *
from generalized_order_attack.utilities import imshow, InputNormalize, get_dataset_model
import torch.multiprocessing as mp
from torch.utils.data import SubsetRandomSampler, DataLoader
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from models.wideresnet import *
from models.resnet import *
import torch.nn.parallel
import time
import numpy as np
import random
from math import pi
import warnings

warnings.filterwarnings('ignore')


def list_type(s):
    return tuple(sorted(map(int, s.split(','))))


parser = argparse.ArgumentParser(
    description='Model Robustness Evaluation')
parser.add_argument('attacks', metavar='attack', type=str, nargs='+',
                    help='attack names')

parser.add_argument('--checkpoint', type=str, default=None, help='path of checkpoint')
parser.add_argument('--stat-dict', type=str, default=None,
                    help='key of stat dict in checkpoint')
parser.add_argument('--arch', type=str, default='resnet50',
                    help='model architecture')
parser.add_argument('--dataset', type=str, default='cifar',
                    help='dataset name')
parser.add_argument('--dataset_path', type=str, default='../data',
                    help='path to datasets directory')
parser.add_argument('--batch_size', type=int, default=100,
                    help='number of examples/minibatch')
parser.add_argument('--parallel', type=int, default=1,
                    help='number of GPUs to train on')
parser.add_argument('--num_batches', type=int, required=False,
                    help='number of batches (default entire dataset)')
parser.add_argument('--per_example', action='store_true', default=False,
                    help='output per-example accuracy')
parser.add_argument('--message', type=str, default="",
                    help='csv message before result')
parser.add_argument('--debug', action='store_true',
                    help='Train Only One Epoch and print training images.')
parser.add_argument('--seed', type=int, default=0, help='RNG seed')

parser.add_argument('--output', type=str, help='output CSV')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

classes_map = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

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

    # if args.multiprocessing_distributed and args.rank % ngpus_per_node != 0:
    #     def print_pass(*args):
    #         pass
    #     builtins.print = print_pass

    model = load_model(args, ngpus_per_node)
    cudnn.benchmark = True

    # setup data loader
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_loader, _ = generate_dataloader('../data', "val", transform_test, workers=args.workers,
                                         batch_size=args.batch_size)

    evaluate(model, test_loader, ngpus_per_node, args)


def generate_dataloader(data, name, transform, workers, batch_size, distributed=False):
    if data is None:
        return None

    if transform is None:
        dataset = datasets.CIFAR10(root=data, train=(name == "train"), download=True,
                                   transform=transforms.ToTensor())
    else:
        dataset = datasets.CIFAR10(root=data, train=(name == "train"), download=True,
                                   transform=transform)

    if name == "train":
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            indices = list(np.random.randint(0, len(dataset), int(len(dataset))))
            train_sampler = SubsetRandomSampler(indices)
    else:
        train_sampler = None

    # Wrap image dataset (defined above) in dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=(train_sampler is None),
                            num_workers=workers,
                            pin_memory=True,
                            sampler=train_sampler)

    return dataloader, train_sampler


def load_model(args, ngpus_per_node):
    # Given Architecture
    if args.arch == 'wideresnet':
        model = WideResNet()
    elif args.arch == 'resnet50':
        if args.stat_dict == 'madry':
            from robustness.datasets import DATASETS
            from robustness.attacker import AttackerModel
            _dataset = DATASETS['cifar']('../data/')
            model = _dataset.get_model(args.arch, False)
            model = AttackerModel(model, _dataset)
            # This should be handle first.
            checkpoint = torch.load(args.checkpoint)
            state_dict_path = 'model'
            if not ('model' in checkpoint):
                state_dict_path = 'state_dict'

            sd = checkpoint[state_dict_path]
            sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            model = model.model
            print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, checkpoint['epoch']))
            print('Natural accuracy --> {}'.format(checkpoint['nat_prec1']))
            print('Robust accuracy --> {}'.format(checkpoint['adv_prec1']))

            # Since Madry's pretrained model only accept normalized tensor,
            # we need to add an layer to normalize before inference.
            model = nn.Sequential(
                InputNormalize(torch.tensor([0.4914, 0.4822, 0.4465]),
                               torch.tensor([0.2023, 0.1994, 0.2010])),
                model
            )
        elif args.stat_dict == 'gat':
            from robustness.datasets import DATASETS
            from robustness.attacker import AttackerModel
            _dataset = DATASETS['cifar']('../data/')
            model = _dataset.get_model(args.arch, False)
            model = AttackerModel(model, _dataset)
            model = nn.Sequential(
                InputNormalize(torch.tensor([0.4914, 0.4822, 0.4465]),
                               torch.tensor([0.2023, 0.1994, 0.2010])),
                model.model
            )
        elif args.stat_dict == 'pat':  # To Debug
            _, model = get_dataset_model(args, dataset_path='../data', dataset_name='cifar')
        elif args.stat_dict == 'gat-fromscratch':
            model = ResNet50()
        else:
            model = ResNet50()
    else:
        print('Model architecture not specified.')
        raise ValueError()

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

    if os.path.exists(args.checkpoint):
        if args.arch == 'wideresnet':
            checkpoint = torch.load(args.checkpoint)
            if args.stat_dict == 'trades':
                sd = {'module.'+k: v for k, v in checkpoint.items()}  # Use this if missing key matching
                model.load_state_dict(sd)
                print("=> loaded checkpoint '{}'".format(args.checkpoint))
            elif args.stat_dict == 'clean':
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        sd = checkpoint['model_state_dict']
                        sd = {'module.'+k: v for k, v in sd.items()}  # Use this if missing key matching
                        model.load_state_dict(sd)
                    else:
                        raise ValueError("Please check State Dict key of checkpoint.")
                    print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, checkpoint['epoch']))
                    print('nat_accuracy --> ', checkpoint['best_acc1'])
            elif args.stat_dict == 'gat' or args.stat_dict == 'gat-fromscratch':
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        sd = checkpoint['model_state_dict']
                        # sd = {k[len('module.'):]: v for k, v in sd.items()}  # Use this if missing key matching
                        # sd = {'module.'+k: v for k, v in sd.items()}  # Use this if missing key matching
                        model.load_state_dict(sd)
                    else:
                        raise ValueError("Please check State Dict key of checkpoint.")
                    print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, checkpoint['epoch']))
                    print('nat_accuracy --> ', checkpoint['best_acc1'])
            elif args.stat_dict == 'gat-previous':
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        sd = checkpoint['model_state_dict']
                        # sd = {k[len('module.'):]: v for k, v in sd.items()}  # Use this if missing key matching
                        if "finetune" not in args.checkpoint:
                            sd = {'module.'+k: v for k, v in sd.items()}  # Use this if missing key matching
                        model.load_state_dict(sd)
                    else:
                        raise ValueError("Please check State Dict key of checkpoint.")
                    print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, checkpoint['epoch']))
                    print('nat_accuracy --> ', checkpoint['best_acc1'])
            else:
                raise ValueError()

        elif args.arch == 'resnet50':
            if args.stat_dict == 'madry':
                pass
            elif args.stat_dict == 'pat':
                pass
            elif args.stat_dict == 'gat':
                checkpoint = torch.load(args.checkpoint)
                if isinstance(checkpoint, dict):
                    state_dict_path = 'model_state_dict'

                    sd = checkpoint[state_dict_path]
                    # sd = {k[len('module.'):]: v for k, v in sd.items()}
                    model.load_state_dict(sd)

                    print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, checkpoint['epoch']))
                    print('nat_accuracy --> ', checkpoint['best_acc1'])
                else:
                    raise ValueError()
            elif args.stat_dict == 'gat-fromscratch':
                checkpoint = torch.load(args.checkpoint)
                if isinstance(checkpoint, dict):
                    state_dict_path = 'model_state_dict'

                    sd = checkpoint[state_dict_path]
                    model.load_state_dict(sd)

                    print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, checkpoint['epoch']))
                    print('nat_accuracy --> ', checkpoint['best_acc1'])
                else:
                    raise ValueError()
            else:
                raise ValueError("State Dict Not Specified.")
        else:
            ValueError("model architecture not specified.")

    return model


def evaluate(model, val_loader, ngpus_per_node, args):
    model.eval()
    print('Cuda available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        model.cuda()

    attack_names: List[str] = args.attacks
    attacks = []
    for attack_name in attack_names:
        tmp = eval(attack_name)
        attacks.append(tmp)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            attacks = [torch.nn.parallel.DistributedDataParallel(attack, device_ids=[args.gpu]) for attack in attacks]
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            attacks = [torch.nn.parallel.DistributedDataParallel(attack) for attack in attacks]
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        attacks = [attack.cuda(args.gpu) for attack in attacks]
    else:
        model = torch.nn.DataParallel(model).cuda()
        attacks = [torch.nn.DataParallel(attack).cuda() for attack in attacks]

    batches_correct: Dict[str, List[torch.Tensor]] = \
        {attack_name: [] for attack_name in attack_names}
    batches_ori_correct: Dict[str, List[torch.Tensor]] = \
        {attack_name: [] for attack_name in attack_names}
    batches_time_used: Dict[str, List[torch.Tensor]] = \
        {attack_name: [] for attack_name in attack_names}

    for batch_index, (inputs, labels) in enumerate(val_loader):
        print(f'BATCH {batch_index:05d}')

        if (
                args.num_batches is not None and
                batch_index >= args.num_batches
        ):
            break

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        file_no = 7
        for attack_name, attack in zip(attack_names, attacks):
            batch_tic = time.perf_counter()
            adv_inputs = attack(inputs, labels)
            if args.debug:
                imshow(inputs, model, classes_map, ground_truth=labels,
                       save_file='images/train/cifar10_eval_'+str(file_no)+' (clean).pdf', show=False)
                imshow(adv_inputs, model, classes_map, ground_truth=labels,
                       save_file='images/train/cifar10_eval_'+str(file_no)+' (attack).pdf', show=False)
                file_no = file_no + 1
            with torch.no_grad():
                ori_logits = model(inputs)
                adv_logits = model(adv_inputs)
            batch_ori_correct = (ori_logits.argmax(1) == labels).detach()
            batch_correct = (adv_logits.argmax(1) == labels).detach()

            batch_accuracy = batch_correct.float().mean().item()
            batch_attack_success_rate = 1.0 - batch_correct[batch_ori_correct].float().mean().item()
            batch_toc = time.perf_counter()
            time_used = torch.tensor(batch_toc - batch_tic)
            print(f'ATTACK {attack_name}',
                  f'accuracy = {batch_accuracy * 100:.1f}',
                  f'attack_success_rate = {batch_attack_success_rate * 100:.1f}',
                  f'time_usage = {time_used:0.2f} s',
                  sep='\t')
            batches_ori_correct[attack_name].append(batch_ori_correct)
            batches_correct[attack_name].append(batch_correct)
            batches_time_used[attack_name].append(time_used)

        if args.debug:
            return

    print('OVERALL')
    accuracies = []
    attack_success_rates = []
    total_time_used = []
    ori_correct: Dict[str, torch.Tensor] = {}
    attacks_correct: Dict[str, torch.Tensor] = {}
    for attack_name in attack_names:
        ori_correct[attack_name] = torch.cat(batches_ori_correct[attack_name])
        attacks_correct[attack_name] = torch.cat(batches_correct[attack_name])
        accuracy = attacks_correct[attack_name].float().mean().item()
        attack_success_rate = 1.0 - attacks_correct[attack_name][ori_correct[attack_name]].float().mean().item()
        time_used = sum(batches_time_used[attack_name]).item()
        print(f'ATTACK {attack_name}',
              f'accuracy = {accuracy * 100:.1f}',
              f'attack_success_rate = {attack_success_rate * 100:.1f}',
              f'time_usage = {time_used:0.2f} s',
              sep='\t')
        accuracies.append(accuracy)
        attack_success_rates.append(attack_success_rate)
        total_time_used.append(time_used)

    with open(args.output, 'a+') as out_file:
        out_csv = csv.writer(out_file)
        out_csv.writerow([args.message])
        out_csv.writerow(['attack_setting'] + attack_names)
        if args.per_example:
            for example_correct in zip(*[
                attacks_correct[attack_name] for attack_name in attack_names
            ]):
                out_csv.writerow(
                    [int(attack_correct.item()) for attack_correct
                     in example_correct])
        out_csv.writerow(['accuracies'] + accuracies)
        out_csv.writerow(['attack_success_rates'] + attack_success_rates)
        out_csv.writerow(['time_usage'] + total_time_used)
        out_csv.writerow(['batch_size', args.batch_size])
        out_csv.writerow(['num_batches', args.num_batches])
        out_csv.writerow([''])


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

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)



if __name__ == '__main__':
    main()
