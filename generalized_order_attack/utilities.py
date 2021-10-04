from typing import Optional, Tuple
import torch
import os
import torchvision.models as torchvision_models
import torchvision
from torch import nn
from robustness.datasets import DATASETS, DataSet
from robustness.model_utils import make_and_restore_model
from robustness.attacker import AttackerModel
from .radar import draw_radar_chart
import matplotlib.pyplot as plt
import numpy as np

from generalized_order_attack.models import CifarResNetFeatureModel, CifarAlexNet, TradesWideResNet
from math import pi
from . import datasets


class MarginLoss(nn.Module):
    """
    Calculates the margin loss max(kappa, (max z_k (x) k != y) - z_y(x)),
    also known as the f6 loss used by the Carlini & Wagner attack.
    """

    def __init__(self, kappa=float('inf'), targeted=False):
        super().__init__()
        self.kappa = kappa
        self.targeted = targeted

    def forward(self, logits, labels):
        correct_logits = torch.gather(logits, 1, labels.view(-1, 1))

        max_2_logits, argmax_2_logits = torch.topk(logits, 2, dim=1)
        top_max, second_max = max_2_logits.chunk(2, dim=1)
        top_argmax, _ = argmax_2_logits.chunk(2, dim=1)
        labels_eq_max = top_argmax.squeeze().eq(labels).float().view(-1, 1)
        labels_ne_max = top_argmax.squeeze().ne(labels).float().view(-1, 1)
        max_incorrect_logits = labels_eq_max * second_max + labels_ne_max * top_max

        if self.targeted:
            return (correct_logits - max_incorrect_logits) \
                .clamp(max=self.kappa).squeeze()
        else:
            return (max_incorrect_logits - correct_logits) \
                .clamp(max=self.kappa).squeeze()


def add_dataset_model_arguments(parser, include_checkpoint=False):
    """
    Adds the argparse arguments to the given parser necessary for calling the
    get_dataset_model command.
    """

    if include_checkpoint:
        parser.add_argument('--checkpoint', type=str, help='checkpoint path')

    parser.add_argument('--arch', type=str, default='resnet50',
                        help='model architecture')
    parser.add_argument('--dataset', type=str, default='cifar',
                        help='dataset name')
    parser.add_argument('--dataset_path', type=str, default='../data',
                        help='path to datasets directory')


def get_dataset_model(
    args=None,
    dataset_path: Optional[str] = None,
    arch: Optional[str] = None,
    checkpoint_fname: Optional[str] = None,
    **kwargs,
) -> Tuple[DataSet, nn.Module]:
    """
    Given an argparse namespace with certain parameters, or those parameters
    as keyword arguments, returns a tuple (dataset, model) with a robustness
    dataset and a FeatureModel.
    """

    if dataset_path is None:
        if args is None:
            dataset_path = '~/datasets'
        else:
            dataset_path = args.dataset_path
    dataset_path = os.path.expandvars(dataset_path)

    dataset_name = kwargs.get('dataset') or args.dataset
    dataset = DATASETS[dataset_name](dataset_path)

    checkpoint_is_feature_model = False

    if checkpoint_fname is None:
        checkpoint_fname = getattr(args, 'checkpoint', None)
    if arch is None:
        arch = args.arch

    if dataset_name.startswith('cifar') and \
            ('resnet' in arch):
        pytorch_pretrained = False
        try:
            model, _ = make_and_restore_model(
                arch=arch,
                dataset=dataset,
                resume_path=checkpoint_fname,
                pytorch_pretrained=pytorch_pretrained,
                parallel=False,
            )
        except RuntimeError as error:  # KeyError
            if 'state_dict' in str(error):
                model, _ = make_and_restore_model(
                    arch=arch,
                    dataset=dataset,
                    parallel=False,
                )
                try:
                    state = torch.load(checkpoint_fname)
                    if 'model' in state:
                        model.model.load_state_dict(state['model'])
                    elif 'model_state_dict' in state:
                        print('Train Epoch: {}'.format(state['epoch']))
                        print('Best Accuracy: {}'.format(state['best_acc1']))
                        model.model.load_state_dict(state['model_state_dict'])
                except RuntimeError as error:
                    if 'state_dict' in str(error):
                        checkpoint_is_feature_model = True
                    else:
                        raise error
            else:
                raise error  # type: ignore
    elif arch == 'trades-wrn':
        model = TradesWideResNet()
        if checkpoint_fname is not None:
            state = torch.load(checkpoint_fname)
            if 'our' in checkpoint_fname:
                model.load_state_dict(state['model_state_dict'])
            else:
                model.load_state_dict(state)
    elif hasattr(torchvision_models, arch):
        if (
            arch == 'alexnet' and
            dataset_name.startswith('cifar') and
            checkpoint_fname != 'pretrained'
        ):
            model = CifarAlexNet(num_classes=dataset.num_classes)
        else:
            if checkpoint_fname == 'pretrained':
                model = getattr(torchvision_models, arch)(pretrained=True)
            else:
                model = getattr(torchvision_models, arch)(
                    num_classes=dataset.num_classes)

        if checkpoint_fname is not None and checkpoint_fname != 'pretrained':
            try:
                state = torch.load(checkpoint_fname)
                model.load_state_dict(state['model'])
            except RuntimeError as error:
                if 'state_dict' in str(error):
                    checkpoint_is_feature_model = True
                else:
                    raise error
    else:
        print(arch, dataset_name, checkpoint_fname)
        raise RuntimeError(f'Unsupported architecture {arch}.')

    if 'resnet' in arch:
        if not isinstance(model, AttackerModel):
            model = AttackerModel(model, dataset)
        if dataset_name.startswith('cifar'):
            model = CifarResNetFeatureModel(model)
        else:
            raise RuntimeError('Unsupported dataset.')
    elif arch == 'trades-wrn':
        pass  # We can't use this as a FeatureModel yet.
    elif arch == 'alexnet':
        pass  # We can't use this as a FeatureModel yet.
    else:
        raise RuntimeError(f'Unsupported architecture {arch}.')

    if checkpoint_is_feature_model:
        if 'our' in checkpoint_fname:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state['model'])

    return dataset, model


def calculate_accuracy(logits, labels):
    correct = logits.argmax(1) == labels
    return correct.float().mean()


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def run_attack_with_random_targets(attack, model, inputs, labels, num_classes):
    """
    Runs an attack with targets randomly selected from all classes besides the
    correct one. The attack should be a function from (inputs, labels) to
    adversarial examples.
    """

    rand_targets = torch.randint(
        0, num_classes - 1, labels.size(),
        dtype=labels.dtype, device=labels.device,
    )
    targets = torch.remainder(labels + rand_targets + 1, num_classes)

    adv_inputs = attack(inputs, targets)
    adv_labels = model(adv_inputs).argmax(1)
    unsuccessful = adv_labels != targets
    adv_inputs[unsuccessful] = inputs[unsuccessful]

    return adv_inputs


def show_attacked_order_and_adv_val(seq, adv_val=None):
    attack_name = ["Hue", "Saturate", "Rotate", "Bright", "Contrast", "L-infinity"]
    s = ""

    if adv_val:
        adv_val[2] = adv_val[2] * 30 if adv_val[2] is not None else None
        for i in range(len(seq)):
            if i == 0:
                s = attack_name[seq[i]] + '(' + str(adv_val[seq[i]]) + ')'
            else:
                s = s + " -> " + attack_name[seq[i]] + '(' + str(adv_val[seq[i]]) + ')'
    else:
        for i in range(len(seq)):
            if i == 0:
                s = attack_name[seq[i]]
            else:
                s = s + " -> " + attack_name[seq[i]]
    return s


def show_attacked_order_by_enable(seq, enabled_attack=None):
    attack_name = ["Hue", "Saturate", "Rotate", "Bright", "Contrast", "L-infinity"]
    s = ""
    for i in range(len(seq)):
        if i == 0:
            s = attack_name[enabled_attack[seq[i]]]
        else:
            s = s + " -> " + attack_name[enabled_attack[seq[i]]]
    return s


def show_enabled_attack(enabled_attack):
    attack_name = ["Hue", "Saturate", "Rotate", "Bright", "Contrast", "L-infinity"]
    s = ""
    for i in range(len(enabled_attack)):
        if i == 0:
            s = attack_name[enabled_attack[i]]
        else:
            s = s + ", " + attack_name[enabled_attack[i]]
    return s


def show_log(attackRecord, final_acc, hue_epsilon, sat_epsilon, rot_epsilon, bright_epsilon, contrast_epsilon,
             linf_epsilon, start_num, iter_num, step_size, multiple_rand_start):
    print('hue: {}, sat: {}, rot: {}, bright: {}, contrast: {}, l_infty: {}, start_num: {}, iter_num: {}, step_size: {}, rand_start: {}, Accuracy: {}'.format(
            hue_epsilon, sat_epsilon, rot_epsilon, bright_epsilon, contrast_epsilon, round(linf_epsilon, 3), start_num,
            iter_num, step_size, multiple_rand_start, round(final_acc, 2)))
    print('---' * 30)


def plot_result(acc, enabled_attack, x_axis_str, x_axis_data, label_str, caption, img_name):
    plt.figure(figsize=(5, 5))
    # x_axis_data = [round(d*1000,1) for d in x_axis_data]
    plt.title("Attack with:  " + str(enabled_attack))
    # for i in range(len(acc)):
    #     plt.plot(start_num, acc, 'o-', label=show_attacked_order(sequence[i]))
    plt.plot(x_axis_data, acc, "o-", label=label_str)
    plt.xlabel(x_axis_str)
    plt.ylabel('Attack Success Rate')
    plt.xticks(x_axis_data)
    plt.ylim(0.0, 1.0)
    if caption is not None:
        plt.text(8, -0.18, caption, horizontalalignment='center', verticalalignment='center')

    for i in range(len(acc)):
        plt.annotate(str(round(acc[i], 2)), (x_axis_data[i], acc[i]), xytext=(x_axis_data[i] + 0.05, acc[i] - 0.01))

    if label_str:
        plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.margins(0.2)
    plt.show()
    plt.savefig('attack_result_' + img_name + '.png', bbox_inches='tight')


def generate_attack_radar(enabled_attack, epsilon_setting):
    adv_val = [(adv_interval[1] - adv_interval[0])/2 for adv_interval in epsilon_setting]
    radar_value = [adv_val[i] if i in enabled_attack else 0 for i in range(6)]
    draw_radar_chart(values=radar_value, legend=show_enabled_attack(enabled_attack))


def imshow(data, model, classes_map, ground_truth=None, save_file='print_grid.pdf', show=False):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3

    model.eval()
    output = model(data)
    data_labels = output.max(1, keepdim=True)[1].cpu()
    ground_truth = ground_truth.cpu() if ground_truth is not None else ground_truth

    for i in range(1, cols * rows + 1):
        img, label = data[i].detach().cpu(), data_labels[i].detach().cpu()
        figure.add_subplot(rows, cols, i)
        if ground_truth is not None:
            plt.title(classes_map[label]+" ("+classes_map[ground_truth[i]]+")")
        else:
            plt.title(classes_map[label])
        plt.axis("off")
        npimg = np.transpose(img.squeeze().numpy(), (1, 2, 0))
        plt.imshow(npimg)

    plt.savefig(save_file)

    if show:
        plt.show()
    print("Figure saved.")


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


def get_imagenet_dict():
    import json
    class_idx = json.load(open("./labels/imagenet_class_index.json"))
    id_to_labels = [class_idx[str(k)][1] for k in range(len(class_idx))]
    nclass_to_id = {class_idx[str(k)][0]: k for k in range(len(class_idx))}

    return id_to_labels, nclass_to_id

