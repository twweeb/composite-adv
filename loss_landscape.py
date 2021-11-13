import argparse
import copy
import numpy as np
import torch
import torchvision
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.font_manager as fm
from generalized_order_attack.attacks import *
from generalized_order_attack.utilities import add_dataset_model_arguments, \
    get_dataset_model, get_imagenet_dict
import kornia
from math import pi
import warnings
warnings.filterwarnings('ignore')
rcParams['font.family'] = 'serif'
rcParams['mathtext.fontset'] = 'cm'
myfont = fm.FontProperties(fname=r'/home/hsiung/labs/NimbusRomNo9L-Reg.otf', size=12) # Font Setting


loss = nn.CrossEntropyLoss()
kornia_pool = [kornia.enhance.adjust_hue, kornia.enhance.adjust_saturation, kornia.geometry.transform.rotate,
              kornia.enhance.adjust_brightness, kornia.enhance.adjust_contrast]

cifar_namelist = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
imagenet_namelist, _ = get_imagenet_dict()


def plot_loss_landscape(model_list, attack, bound, data_loader, start_num, iter_num, plot_type='all',
                        sample=1, points=200, one_page=None,
                        pgd_area_grain=200, early_stop=False,
                        output_dir='results/loss_landscape/', msg='hue', args_dict=None, live_plot=False):
    def sampling_loss(_model, _images, _labels):
        _x_axis = [((bound[1] - bound[0]) * i / points) + bound[0] for i in range(points + 1)]
        _y_axis = []
        _predict = torch.ones(points + 1).cuda()
        for val_i in range(points + 1):
            adv_img = kornia_pool[attack](_images, torch.tensor(_x_axis[val_i]).cuda())
            _outputs = _model(adv_img)

            _y_axis.append(loss(_outputs, _labels).unsqueeze(0).detach().cpu().numpy())
#             cur_pred = _outputs.max(1, keepdim=True)[1]
            if _outputs.argmax(1) != _labels:
                _predict[val_i] = 0
        return _x_axis, _y_axis, _predict

    def record_loss_trace(_model, _images, _labels):
        grain_t = torch.tensor(pgd_area_grain)
        bound_t = torch.tensor(bound)
        start_points = [((2 * i + 1.) / start_num) * (bound_t[1] - bound_t[0]) / 2 + bound_t[0] for i in range(start_num)]
        _predict = torch.zeros(grain_t)
        _loss_rec = []

        def record_loss(_start_val):
            _step_size = (bound_t[1] - bound_t[0]) * 0.05
            _adv_val = _start_val.requires_grad_()
            _part_loss_rec = []

            _adv_img = kornia_pool[attack](_images, _adv_val)
            for i in range(iter_num):
                _outputs = _model(_adv_img)
                _model.zero_grad()
                _cost = loss(_outputs, _labels)
                _part_loss_rec.append([_adv_val.item(), _cost.item()])

#                 cur_pred = _outputs.max(1, keepdim=True)[1]
                _predict_idx = ((_adv_val - bound_t[0]) / (bound_t[1] - bound_t[0]) * grain_t).to(torch.int32)
                if _outputs.argmax(1) != _labels:
                    if _predict_idx == grain_t:
                        _predict_idx = -1
                    _predict[_predict_idx] = 1.0
                    if early_stop:
                        break

                _adv_val_grad = torch.autograd.grad(_cost, _adv_val, retain_graph=False)[0]
                _adv_val = torch.clamp(_adv_val + torch.sign(_adv_val_grad) * _step_size, bound_t[0],
                                       bound_t[1]).detach().requires_grad_()
                _adv_img = kornia_pool[attack](_images, _adv_val)

            # record the last update
            _part_loss_rec.append([_adv_val.item(), loss(_model(_adv_img), _labels).item()])

            return _part_loss_rec

        for i in range(len(start_points)):
            part_loss_rec = record_loss(start_points[i])
            if len(part_loss_rec) == 0:
                continue
            else:
                _loss_rec.append(copy.deepcopy(part_loss_rec))
        return _loss_rec, _predict

    def plot_loss_traces():
        fig, subfig = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 5.5]})
        fig.subplots_adjust(top=0.98, bottom=0.15, left=0.23, right=0.85)

        subfig[0].imshow(np.transpose(sample_img, (1, 2, 0)))
        subfig[0].axis('off')

        ax = plt.gca()
        # subfig[1].set_title("Loss Traces During PGD Process (■ → ▲)")
        for trial in loss_rec:
            t = torch.tensor(trial)
            color = next(ax._get_lines.prop_cycler)['color']
            subfig[1].plot(t[..., 0][-1], t[..., 1][-1], linestyle='', markeredgecolor='none', marker='^', color=color,
                           alpha=0.8)
            subfig[1].plot(t[..., 0][0], t[..., 1][0], linestyle='', markeredgecolor='none', marker='s', color=color,
                           alpha=0.8)
            subfig[1].plot(t[..., 0], t[..., 1], linestyle='-', color=color, alpha=0.8)

        if pgd_wrong_area_list[0] is not None:
            x_axis_list_adjust = [((bound[1] - bound[0]) * i / pgd_area_grain) + bound[0] for i in range(pgd_area_grain)]
            high_light = x_axis_list_adjust[pgd_wrong_area_list[0]].detach().cpu().numpy()

            l_i = 0

            span_left = span_right = 0
            span_interval = (bound[1] - bound[0]) / pgd_area_grain
            for x in high_light:
                if span_right == 0:
                    span_left = x
                elif x - span_right > 0.5 * span_interval or x == high_light[-1]:
                    subfig[1].axvspan(span_left, span_right,
                                      color="crimson", alpha=0.3, lw=0, label="_" * l_i + "Predict Error")
                    span_left = x
                    l_i = l_i + 1
                span_right = x + span_interval
                if x == high_light[-1]:
                    subfig[1].axvspan(span_left, span_right,
                                      color="crimson", alpha=0.3, lw=0, label="_" * l_i + "Predict Error")
        if attack == 0:
            legend_str = '\n'.join(["■ start", "▲  end"])
            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            subfig[1].text(0.05, 0.95, legend_str, transform=ax.transAxes, fontsize=12, va='top', bbox=props)

        subfig[1].set_xticks([round((bound[1] - bound[0]) * i / 4 + bound[0], 2) for i in range(5)])
        if attack == 0:
            subfig[1].set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(1, 2), useOffset=False)
        plt.locator_params(axis='y', tight=True, nbins=5)
        # subfig[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        if attack == 2:
            plt.xlabel(msg.capitalize() + ' degree', fontsize=18)
            # fig.text(0.5, 0, msg.capitalize() + ' degree', ha='center', rotation='horizontal', fontsize=20)
        else:
            plt.xlabel(msg.capitalize() + ' value', fontsize=18)
            # fig.text(0.5, 0, msg.capitalize() + ' value', ha='center', rotation='horizontal', fontsize=20)
        plt.ylabel('Xent Loss', fontsize=18)
        # fig.text(0.05, 0.4, 'Xent Loss', va='center', rotation='vertical', fontsize=20)
        pdf.savefig()
        plt.close()

    def plot_separately():
        fig, subfig = plt.subplots(len(y_axis_list) + 1, 1)
        fig.subplots_adjust(top=0.98, bottom=0.13, left=0.18, right=0.89, hspace=0.4)
        subfig[0].imshow(np.transpose(sample_img, (1, 2, 0)))
        subfig[0].axis('off')

        axs_idx = 1
        for i in range(len(y_axis_list)):
            subfig[axs_idx].plot(x_axis_list[i].detach().cpu().numpy(),
                                 y_axis_list[i].detach().cpu().numpy(), '-')
            subfig[axs_idx].text(0.5, 0.5, model_name_list[i], transform=subfig[axs_idx].transAxes,
                                 fontsize=12, color='gray', alpha=0.3,
                                 ha='center', va='center')

            # plt.plot(x_axis.detach().cpu().numpy(), predict.detach().cpu().numpy(), '-')
            if wrong_area_list[i] is not None:
                high_light = x_axis_list[i][wrong_area_list[i]].detach().cpu().numpy()
                l_i = 0

                span_left = span_right = 0
                span_interval = (bound[1] - bound[0]) / points
                for x in high_light:
                    if span_right == 0:
                        span_left = x
                    elif x - span_right > 0.5 * span_interval or x == high_light[-1]:
                        subfig[axs_idx].axvspan(span_left, span_right,
                                                color="crimson", alpha=0.3, lw=0, label="_" * l_i + "Predict Error")

                        # if show_legend:
                        #     handles, leg = axs[i + 1].get_legend_handles_labels()
                        #     fig.legend(handles, leg, loc='lower right')
                        #     show_legend = False
                        span_left = x
                        l_i = l_i + 1
                    span_right = x + span_interval
            plt.yticks(fontsize=15)
            plt.ticklabel_format(axis='y', style='sci', scilimits=(1, 2), useOffset=False)
            plt.locator_params(axis='y', tight=True, nbins=3)
            # subfig[axs_idx].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            if i != len(y_axis_list) - 1:
                subfig[axs_idx].xaxis.set_visible(False)
            # axs[axs_idx].yaxis.set_visible(False)
            axs_idx = axs_idx + 1

        plt.xticks([round((bound[1] - bound[0]) * i / 4 + bound[0], 2) for i in range(5)], fontsize=15)
        # plt.xlabel(msg.capitalize() + ' Value', fontsize=13)
        fig.text(0.05, 0.4, 'CrossEntropy', va='center', rotation='vertical', fontsize=13)
        fig.text(0.5, 0.02, msg.capitalize() + ' Value', ha='center', rotation='horizontal', fontsize=13)
        pdf.savefig()
        plt.close()

    def plot_together():
        fig, subfig = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1.8, 6]})
        fig.subplots_adjust(top=0.98, bottom=0.09, left=0.23, right=0.85, wspace=0, hspace=0.12)

        subfig[0].imshow(np.transpose(sample_img, (1, 2, 0)), aspect='auto')
        subfig[0].axis('off')
        # Draw Loss landscape together
        for i in range(len(y_axis_list)):
            subfig[1].plot(x_axis_list[i], y_axis_list[i], '-',
                           label=model_name_list[i])
            if attack == 0:
                subfig[1].legend(loc='upper left', prop=myfont) #{'size': 12}

        subfig[1].set_xticks([round((bound[1] - bound[0]) * i / 4 + bound[0], 2) for i in range(5)])
        if attack == 0:
            subfig[1].set_xticklabels(['$-$π', '$-$π/$2$', '$0$', 'π/$2$', 'π'])
        elif attack == 1:
            subfig[1].set_xticklabels(['$0.7$', '$0.85$', '$1.0$', '$1.15$', '$1.3$'])
        elif attack == 2:
            subfig[1].set_xticklabels(['$-10\degree$', '$-5\degree$', '$0\degree$', '$5\degree$', '$10\degree$'])
        elif attack == 3:
            subfig[1].set_xticklabels(['$-0.2$', '$-0.1$', '$0$', '$0.1$', '$0.2$'])
        elif attack == 4:
            subfig[1].set_xticklabels(['$0.7$', '$0.85$', '$1.0$', '$1.15$', '$1.3$'])
        plt.xticks(fontsize=20, fontproperties=myfont)
        plt.yticks(fontsize=20, fontproperties=myfont)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useOffset=False, useMathText=True)
        plt.locator_params(axis='y', tight=True, nbins=5)
        # subfig[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # plt.xlabel(msg.capitalize() + ' Value', fontsize=13)

#         if attack == 2:
#             plt.xlabel(msg.capitalize(), fontsize=18, fontproperties=myfont) # + ' degree'
#             # fig.text(0.5, 0, msg.capitalize() + ' degree', ha='center', rotation='horizontal', fontsize=20)
#         else:
#             plt.xlabel(msg.capitalize(), fontsize=18, fontproperties=myfont) # + ' value'
#             # fig.text(0.5, 0, msg.capitalize() + ' value', ha='center', rotation='horizontal', fontsize=20)

#         if attack != 0:
#             plt.ylabel('Xent Loss', fontsize=18, fontproperties=myfont)
        
        # fig.text(0.05, 0.4, 'Xent Loss', va='center', rotation='vertical', fontsize=20)
        pdf.savefig()
        if live_plot:
            print("id:", idx, imagenet_namelist[target.cpu().item()])
            plt.show()
        plt.close()

    with PdfPages(output_dir + msg + '_' + plot_type + '.pdf') as pdf:
        for idx, (data, target) in enumerate(data_loader):
            if one_page is None:
                if idx == sample:
                    break
            else:
                if idx < one_page:
                    continue
                elif idx > one_page:
                    break

            data = data.cuda()
            target = target.cuda()
            x_axis_list = []
            y_axis_list = []
            predict_list, wrong_area_list = [], []
            pgd_predict_list, pgd_wrong_area_list = [], []
            model_name_list = []
            loss_rec = []
            if (plot_type == 'all' or plot_type == 'trace') and 'normal' in model_list:
                loss_rec, predict = record_loss_trace(model_list['normal'], data, target)
                pgd_predict_list.append(predict)
                pgd_wrong_area_list.append((predict == 1.) if predict.max() != 0. else None)
            for _md in model_list:
                x_axis, y_axis, predict = sampling_loss(model_list[_md], data, target)
                x_axis_list.append(x_axis)
                y_axis_list.append(y_axis)
                predict_list.append(predict)
                if args_dict is not None:
                    model_name_list.append(args_dict[_md].display_text)
                else:
                    model_name_list.append(model_list[_md].display_text)
                wrong_area_list.append((predict == 1.) if predict.max() != 0. else None)

            # show the image corresponding to generated loss landscape
            sample_img = torchvision.utils.make_grid(
                [kornia_pool[attack](data, torch.tensor(bound[0] + i * (bound[1] - bound[0]) / 4).cuda())[0].cpu() 
                 for i in range(5)]
            ).numpy()
            if plot_type == 'all' or plot_type == 'together':
                plot_together()

            if plot_type == 'all' or plot_type == 'separate':
                plot_separately()

            if plot_type == 'all' or plot_type == 'trace':
                plot_loss_traces()

        # Set the file's metadata.
        d = pdf.infodict()
        d['Author'] = 'Lei Hsiung'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Adversarial example generation')

    add_dataset_model_arguments(parser, include_checkpoint=True)

    parser.add_argument('--batch_index', type=int, default=0,
                        help='batch index to generate adversarial examples '
                             'for')
    parser.add_argument('--sample', type=int, default=5,
                        help='number of samples to generate')
    parser.add_argument('--shuffle', default=False, action='store_true',
                        help="Shuffle dataset before choosing a batch")
    parser.add_argument('--output', type=str,
                        help='output PDF file directory')
    parser.add_argument('--attack_id', type=int, choices=[0, 1, 2, 3, 4],
                        help='threat model id')
    parser.add_argument('--plot_type', type=str, choices=['all', 'separate', 'together', 'trace'],
                        help='type of loss landscape to plot')
    parser.add_argument('--msg', type=str,
                        help='model message')
    parser.add_argument('--random_seed', type=int, default=None,
                        help='seed for the Torch RNG')

    args = parser.parse_args()

    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)

    wrn_checkpoint_dict = {'Normal$^{*}$': 'pretrained_model/our-wideresnet34-10-clean.pt',
                           'Trades$_{\ell_{\infty}}^{*}$': 'pretrained_model/cifar_trades_wideresnet34_10_linf_8.pt',
                           'GAT-s$^{*}$': 'pretrained_model/our-all-linf10-random-ep70.pt',
                           'GAT-f$^{*}$': 'pretrained_model/our-all-linf10-random-finetune-18.pt',
                           }
    models = dict()
    dataset = None
    for checkpoint in wrn_checkpoint_dict:
        args.arch = 'trades-wrn'
        args.checkpoint = wrn_checkpoint_dict[checkpoint]
        if dataset is None:
            dataset, md = get_dataset_model(args)
        else:
            _, md = get_dataset_model(args)
        models[checkpoint] = md

    _, test_loader = dataset.make_loaders(workers=8, batch_size=1, only_val=True, shuffle_val=args.shuffle)
    for model_name in models:
        models[model_name].eval()
        if torch.cuda.is_available():
            print(model_name, 'is in cuda.')
            models[model_name].cuda()

    attack_bound = [(-pi, pi), (0.7, 1.3), (-10, 10), (-0.2, 0.2), (0.7, 1.3)][args.attack_id]

    plot_loss_landscape(models, args.attack_id, attack_bound, test_loader,
                        start_num=20, iter_num=5, plot_type=args.plot_type,
                        sample=args.sample, points=256,
                        pgd_area_grain=50, early_stop=False,
                        output_dir=args.output, msg=args.msg)
