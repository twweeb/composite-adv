import torch
import os
import time
import numpy as np
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler, DataLoader


def make_dataloader(dataset_path, dataset_name, batch_size, transform=None, train=False, distributed=False, shuffle=False):
    if dataset_path is None:
        return None

    # setup data loader
    if 'cifar10' in dataset_name:
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        dataset = datasets.CIFAR10(root=dataset_path, train=train, download=True, transform=transform)
    elif 'imagenet' in dataset_name:
        if transform is None:
            from data_augmentation import TEST_TRANSFORMS_IMAGENET
            transform = TEST_TRANSFORMS_IMAGENET
        dataset = datasets.ImageFolder(dataset_path, transform=transform)
    else:
        raise ValueError()

    if train:
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            indices = list(np.random.randint(0, len(dataset), int(len(dataset))))
            train_sampler = SubsetRandomSampler(indices)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, sampler=train_sampler, pin_memory=True)
        return dataloader, train_sampler

    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=shuffle, pin_memory=True)
        return dataloader


def make_model(arch, dataset_name, checkpoint_path=None):
    # Given Architecture
    if dataset_name == 'cifar10':
        from . import cifar10_models
        if arch == 'wideresnet':
            model = cifar10_models.wideresnet.WideResNet()
        elif arch == 'resnet50':
            model = cifar10_models.resnet.ResNet50()
        else:
            raise ValueError('Model architecture not specified.')
    elif dataset_name == 'imagenet':
        if arch == 'resnet50':
            from torchvision.models import resnet50
            model = resnet50()
        else:
            raise ValueError('Model architecture not specified.')
    else:
        raise ValueError('Dataset not specified.')

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)  # pickle_module=dill

        if isinstance(checkpoint, dict):
            state_dict_path = 'model'
            if not ('model' in checkpoint):
                if 'state_dict' in checkpoint:
                    state_dict_path = 'state_dict'
                elif 'model_state_dict' in checkpoint:
                    state_dict_path = 'model_state_dict'
                else:
                    raise ValueError("Please check State Dict key of checkpoint.")

            sd = checkpoint[state_dict_path]
            # sd = {k[len('module.'):]: v for k, v in sd.items()}  # Use this if missing key matching
            # sd = {k[len('1.'):]: v for k, v in sd.items() if k.startswith('1.')}  # Debug
            # sd = {'module.'+k: v for k, v in sd.items()}  # Use this if missing key matching
            model.load_state_dict(sd)
            print("=> loaded checkpoint '{}'".format(checkpoint_path))
            print('nat_accuracy --> ', checkpoint['best_acc1'])
        else:
            ValueError("Checkpoint is not dict type.")

    return model


def make_madry_model(arch, dataset_name, checkpoint_path=None):
    from robustness.datasets import DATASETS
    from robustness.attacker import AttackerModel
    _dataset = DATASETS['cifar']()
    model = _dataset.get_model(arch, False)
    model = AttackerModel(model, _dataset)
    checkpoint = torch.load(checkpoint_path)

    state_dict_path = 'model'
    if not ('model' in checkpoint):
        state_dict_path = 'state_dict'

    sd = checkpoint[state_dict_path]
    sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model = model.model
    print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    print('Natural accuracy --> {}'.format(checkpoint['nat_prec1']))
    print('Robust accuracy --> {}'.format(checkpoint['adv_prec1']))

    return model


def make_trades_model(arch, dataset_name, checkpoint_path=None):
    from cifar10_models.wideresnet import WideResNet
    model = WideResNet()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    print("=> loaded checkpoint '{}'".format(checkpoint_path))

    return model


def make_pat_model(arch, dataset_name, checkpoint_path=None):
    pass


def robustness_evaluate(model, threat_model, val_loader):
    model.eval()

    batches_ori_correct, batches_correct, batches_time_used = [], [], []
    for batch_index, (inputs, labels) in enumerate(val_loader):
        print(f'BATCH {batch_index:05d}')

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        batch_tic = time.perf_counter()
        adv_inputs = threat_model(inputs, labels)
        with torch.no_grad():
            ori_logits = model(inputs)
            adv_logits = model(adv_inputs)
        batch_ori_correct = (ori_logits.argmax(1) == labels).detach()
        batch_correct = (adv_logits.argmax(1) == labels).detach()

        batch_accuracy = batch_correct.float().mean().item()
        batch_attack_success_rate = 1.0 - batch_correct[batch_ori_correct].float().mean().item()
        batch_toc = time.perf_counter()
        time_used = torch.tensor(batch_toc - batch_tic)
        print(f'accuracy = {batch_accuracy * 100:.2f}',
              f'attack_success_rate = {batch_attack_success_rate * 100:.2f}',
              f'time_usage = {time_used:0.2f} s',
              sep='\t')
        batches_ori_correct.append(batch_ori_correct)
        batches_correct.append(batch_correct)
        batches_time_used.append(time_used)

    print('OVERALL')
    accuracies = []
    attack_success_rates = []
    total_time_used = []

    ori_correct = torch.cat(batches_ori_correct)
    attacks_correct = torch.cat(batches_correct)
    accuracy = attacks_correct.float().mean().item()
    attack_success_rate = 1.0 - attacks_correct[ori_correct].float().mean().item()
    time_used = sum(batches_time_used).item()
    print(f'accuracy = {accuracy * 100:.5f}',
          f'attack_success_rate = {attack_success_rate * 100:.5f}',
          f'time_usage = {time_used:0.2f} s',
          sep='\t')
    accuracies.append(accuracy)
    attack_success_rates.append(attack_success_rate)
    total_time_used.append(time_used)


class InputNormalize(nn.Module):
    def __init__(self, new_mean, new_std):
        super(InputNormalize, self).__init__()
        new_std = new_std[..., None, None]
        new_mean = new_mean[..., None, None]

        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        x_normalized = (x - self.new_mean)/self.new_std
        return x_normalized


class EvalModel(nn.Module):
    def __init__(self, model, normalize_param=None, input_normalized=False):
        super(EvalModel, self).__init__()
        if input_normalized:
            if normalize_param is None:
                raise ValueError
            self.normalizer = InputNormalize(torch.tensor(normalize_param['mean']),
                                             torch.tensor(normalize_param['std']))
        self.input_normalized = input_normalized
        self.model = model

    def forward(self, inp):
        if self.input_normalized:
            normalized_inp = self.normalizer(inp)
            output = self.model(normalized_inp)
        else:
            output = self.model(inp)

        return output

