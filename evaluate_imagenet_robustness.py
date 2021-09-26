import argparse
import csv
import os
import random
import time
import warnings
from typing import Dict, List
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50
from models.wideresnet import *
from generalized_order_attack.attacks import *

warnings.filterwarnings('ignore')


def list_type(s):
    return tuple(sorted(map(int, s.split(','))))


parser = argparse.ArgumentParser(
    description='Model Robustness Evaluation')
parser.add_argument('attacks', metavar='attack', type=str, nargs='+',
                    help='attack names')

parser.add_argument('--checkpoint', type=str, default=None,
                    help='checkpoint path')
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
parser.add_argument('--seed', type=int, default=0, help='RNG seed')

parser.add_argument('--output', type=str, help='output CSV')

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


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def forward(self, input_tensor):
        return self.normalize(input_tensor)


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

    model = load_model(args)
    cudnn.benchmark = True

    # setup data loader
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    DATA_DIR = '/work/hsiung1024/imagenet'  # Original images come in shapes of [3,64,64]
    # Define training and validation data paths
    VALID_DIR = os.path.join(DATA_DIR, 'val')

    test_loader = generate_dataloader(VALID_DIR, "val", transform_test, workers=ngpus_per_node * 4,
                                      batch_size=args.batch_size)

    evaluate(model, test_loader, ngpus_per_node, args)


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


def load_model(args):
    if args.arch == 'wideresnet':
        model = WideResNet(num_classes=200).cuda()
    elif args.arch == 'resnet50':
        if args.checkpoint is not None:
            from robustness.model_utils import make_and_restore_model
            from robustness.datasets import DATASETS
            _dataset = DATASETS['imagenet']('/work/hsiung1024/imagenet')
            model, _ = make_and_restore_model(arch=args.arch,
                                              dataset=_dataset, resume_path=args.checkpoint)
            model = model.model
        else:
            model = resnet50(pretrained=True)
    else:
        print('Model architecture not specified.')

    model = nn.Sequential(
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        model
    )
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

        for attack_name, attack in zip(attack_names, attacks):
            batch_tic = time.perf_counter()
            adv_inputs = attack(inputs, labels)
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
        # Make sure we can reproduce the testing result.
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        # cudnn.deterministic = True

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


if __name__ == '__main__':
    main()
