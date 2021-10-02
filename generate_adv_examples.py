"""
Scripts that generates a number of adversarial examples for each of several
attacks against a particular network.
"""
import argparse
import itertools
import random
import os
import numpy as np
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50
from models.wideresnet import *
from models.resnet import *
from generalized_order_attack.attacks import *
from generalized_order_attack.utilities import InputNormalize, get_dataset_model
import warnings
from math import pi
warnings.filterwarnings('ignore')


def tile_images(images):
    """
    Given a numpy array of shape r x c x C x W x H, where r and c are rows and
    columns in a grid of images, tiles the images into a numpy array
    C x (W * c) x (H * r).
    """

    return np.concatenate(np.concatenate(images, axis=2), axis=2)


parser = argparse.ArgumentParser(
    description='Adversarial example generation')

parser.add_argument('attacks', metavar='attack', type=str, nargs='+',
                    help='attack names')
parser.add_argument('--checkpoint', type=str, help='checkpoint path')
parser.add_argument('--arch', type=str, default='resnet50',
                    help='model architecture')
parser.add_argument('--stat-dict', type=str, default=None,
                    help='key of stat dict in checkpoint')
parser.add_argument('--dataset', type=str, default='cifar',
                    help='dataset name')
parser.add_argument('--dataset_path', type=str, default='../data',
                    help='path to datasets directory')
parser.add_argument('--batch_size', type=int, default=16,
                    help='number of examples to generate '
                         'adversarial examples for')
parser.add_argument('--batch_index', type=int, default=0,
                    help='batch index to generate adversarial examples '
                         'for')
parser.add_argument('--shuffle', default=False, action='store_true',
                    help="Shuffle dataset before choosing a batch")
parser.add_argument('--layout', type=str, default='vertical',
                    help='lay out the same images on the same row '
                         '(horizontal) or column (vertical)')
parser.add_argument('--robust_num', type=int, default=0,
                    help='numbers of attacks use unsuccessful examples')
parser.add_argument('--only_successful', action='store_true',
                    default=False,
                    help='only show images where adversarial example '
                         'was generated for all attacks')
parser.add_argument('--seed', type=int, default=0, help='RNG seed')
parser.add_argument('--output', type=str,
                    help='output PNG file')

cifar_namelist = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def generate_dataloader(data, dataset_name, shuffle, workers, batch_size):
    if data is None:
        return None

    # setup data loader
    if 'cifar' in dataset_name:
        dataset = datasets.CIFAR10(root=data, train=False, download=False, transform=transforms.ToTensor())
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        dataset = datasets.ImageFolder(data, transform=transform)

    # Wrap image dataset (defined above) in dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=workers,
                            pin_memory=True)

    return dataloader


def load_imagenet_model(args):
    if args.arch == 'wideresnet':
        _model = WideResNet(num_classes=200).cuda()
    elif args.arch == 'resnet50':
        _model = resnet50(pretrained=True)
    else:
        print('Model architecture not specified.')
        raise ValueError()

    _model = nn.Sequential(
        InputNormalize(torch.tensor([0.485, 0.456, 0.406]),
                       torch.tensor([0.229, 0.224, 0.225])),
        _model
    )
    return _model


def load_cifar10_model(args):
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
    else:
        model = torch.nn.DataParallel(model).cuda()

    if os.path.exists(args.checkpoint):
        if args.arch == 'wideresnet':
            checkpoint = torch.load(args.checkpoint)
            if args.stat_dict == 'trades':
                sd = {'module.'+k: v for k, v in checkpoint.items()}  # Use this if missing key matching
                model.load_state_dict(sd)
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


def print_adv_examples(model, val_loader, args):
    inputs, labels = next(itertools.islice(
        val_loader, args.batch_index, None))
    if torch.cuda.is_available():
        inputs, labels = inputs.cuda(), labels.cuda()
    N, C, H, W = inputs.size()

    attacks = [None] + args.attacks
    out_advs = np.ones((len(attacks), N, C, H, W))
    out_diffs = np.ones_like(out_advs)

    orig_labels = model(inputs).argmax(1)
    all_successful = np.ones(N, dtype=bool)
    all_labels = np.zeros((len(attacks), len(orig_labels)), dtype=int)
    all_labels[0] = orig_labels.cpu().detach().numpy()

    for attack_index, attack_name in enumerate(attacks):
        print(f'generating examples for {attack_name or "no"} attack')

        attack_params = None
        if attack_name is None:
            out_advs[attack_index] = inputs.cpu().numpy()
            out_diffs[attack_index] = 0
        else:
            attack = eval(attack_name)

            advs = attack(inputs, labels)
            adv_labels = model(advs).argmax(1)
            unsuccessful = (adv_labels == labels).cpu().detach().numpy()\
                .astype(bool)
            successful = (adv_labels != labels).cpu().detach().numpy() \
                .astype(bool)

            print(f'accuracy = {np.mean(1 - successful) * 100:.1f}')
            diff = (advs - inputs).cpu().detach().numpy()
            advs = advs.cpu().detach().numpy()
            if attack_index < args.robust_num:
                out_advs[attack_index, unsuccessful] = advs[unsuccessful]
                out_diffs[attack_index, unsuccessful] = diff[unsuccessful]
                all_successful[successful] = False
            else:
                out_advs[attack_index, successful] = advs[successful]
                out_diffs[attack_index, successful] = diff[successful]
                all_successful[unsuccessful] = False

            all_labels[attack_index] = adv_labels.cpu().detach().numpy()

            # mark examples that changed by less than 1/1000 as not successful
            all_successful[np.all(np.abs(diff) < 1e-3,
                                  axis=(1, 2, 3))] = False

    if args.only_successful:
        out_advs = out_advs[:, all_successful]
        out_diffs = out_diffs[:, all_successful]
        all_labels = all_labels[:, all_successful]

    out_diffs = np.clip(out_diffs * 3 + 0.5, 0, 1)

    combined_image: np.ndarray
    if args.layout == 'vertical':
        if len(attacks) == 2:
            combined_grid = np.concatenate([
                out_advs,
                np.clip(out_diffs[1:2], 0, 1),
            ], axis=0)
        else:
            combined_grid = np.concatenate([
                out_advs,
                np.ones((len(attacks), 1, C, H, W)),
                out_diffs,
            ], axis=1)
        combined_image = tile_images(combined_grid)
    elif args.layout == 'horizontal_alternate':
        rows = []
        for i in range(out_advs.shape[1]):
            row = [out_advs[0, i]]
            for adv, diff in zip(out_advs[1:, i], out_diffs[1:, i]):
                row.append(np.ones((C, H, W // 4)))
                row.append(adv)
                row.append(diff)
            rows.append(np.concatenate(row, axis=2))
        combined_image = np.concatenate(rows, axis=1)
    elif args.layout == 'vertical_alternate':
        rows = []
        for i in range(out_advs.shape[0]):
            row = []
            for adv, diff in zip(out_advs[i], out_diffs[i]):
                row.append(np.ones((C, H, W // 4)))
                row.append(adv)
                row.append(diff)
            rows.append(np.concatenate(row[1:], axis=2))
        combined_image = np.concatenate(rows, axis=1)
    else:
        raise ValueError(f'Unknown layout "{args.layout}"')
    save_image(torch.from_numpy(combined_image), args.output)
    return True


def main():
    args = parser.parse_args()

    if args.seed is not None and not args.shuffle:
        # Make sure we can reproduce the testing result.
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if 'cifar' in args.dataset:
        model = load_cifar10_model(args)
    else:
        model = load_imagenet_model(args)
    model.eval()
    cudnn.benchmark = True

    if 'cifar' in args.dataset:
        val_loader = generate_dataloader('../data', args.dataset, args.shuffle,
                                         workers=4, batch_size=args.batch_size)
    else:
        val_loader = generate_dataloader('/work/hsiung1024/imagenet/val', args.dataset, args.shuffle,
                                         workers=4, batch_size=args.batch_size)

    examples_found = False
    while not examples_found:
        try:
            examples_found = print_adv_examples(model, val_loader, args)
        except ValueError:
            examples_found = False


if __name__ == '__main__':
    main()
