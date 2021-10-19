import torch
import torch.nn as nn
from math import pi
from . import sinkhorn
import kornia
import torch.nn.functional as F


class CompositeAttack(nn.Module):
    """
    Base class for attacks using the composite attack..
    """

    def __init__(self, model, enabled_attack, mode='eval', local_rank=-1,
                 hue_epsilon=None, sat_epsilon=None, rot_epsilon=None,
                 bright_epsilon=None, contrast_epsilon=None, linf_epsilon=None,
                 attack_power='strong',
                 start_num=1, iter_num=5, inner_iter_num=10, multiple_rand_start=True, order_schedule='random'):
        super().__init__()
        self.model = model
        self.local_rank = local_rank
        self.device = 'cuda' if local_rank == -1 else 'cuda:'+str(local_rank)
        self.fixed_order = enabled_attack
        self.enabled_attack = tuple(sorted(enabled_attack))
        self.mode = mode
        self.seq_num = len(enabled_attack)  # attack_num
        self.attack_pool_eval = (
            self.eval_hue, self.eval_sat, self.eval_rot, self.eval_bright, self.eval_contrast, self.eval_linf)
        self.attack_pool_train = (
            self.train_hue, self.train_sat, self.train_rot, self.train_bright, self.train_contrast, self.train_linf)
        if mode == 'eval':
            self.attack_dict = tuple([self.attack_pool_eval[i] for i in self.enabled_attack])
        elif mode == 'train' or mode == 'fast_train':
            self.attack_dict = tuple([self.attack_pool_train[i] for i in self.enabled_attack])
        else:
            ValueError()
        self.linf_idx = self.enabled_attack.index(5) if 5 in self.enabled_attack else None

        self.eps_pool_weak = torch.tensor(
            [(-pi / 2, pi / 2), (0.8, 1.2), (-5, 5), (-0.1, 0.1), (0.8, 1.2), (-4 / 255, 4 / 255)])
        self.eps_pool_strong = torch.tensor(
            [(-pi, pi), (0.7, 1.3), (-10, 10), (-0.2, 0.2), (0.7, 1.3), (-8 / 255, 8 / 255)])
        self.eps_pool_tough = torch.tensor(
            [(-pi, pi), (0.0, 3.0), (-30, 30), (-0.3, 0.3), (0.50, 2.0), (-8 / 255, 8 / 255)])
        self.eps_pool_im_strong = torch.tensor(
            [(-pi, pi), (0.7, 1.3), (-10, 10), (-0.2, 0.2), (0.7, 1.3), (-4 / 255, 4 / 255)])
        self.eps_pool_im_tough = torch.tensor(
            [(-pi, pi), (0.0, 3.0), (-30, 30), (-0.3, 0.3), (0.50, 2.0), (-4 / 255, 4 / 255)])
        self.eps_pool_custom = [hue_epsilon, sat_epsilon, rot_epsilon, bright_epsilon, contrast_epsilon, linf_epsilon]

        if attack_power == 'weak':
            self.eps_pool = self.eps_pool_weak
        elif attack_power == 'strong':
            self.eps_pool = self.eps_pool_strong
        elif attack_power == 'tough':
            self.eps_pool = self.eps_pool_tough
        elif attack_power == 'im_strong':
            self.eps_pool = self.eps_pool_im_strong
        elif attack_power == 'im_tough':  # Used for generating adversarial examples.
            self.eps_pool = self.eps_pool_im_tough
        else:
            print("Does not specify attack power, using default:", "weak")
        for i in range(6):
            if self.eps_pool_custom[i] is not None:
                self.eps_pool[i] = torch.tensor(self.eps_pool_custom[i])

        if order_schedule not in ('fixed', 'random', 'scheduled'):
            print("order_schedule: {}, should be either 'fixed', 'random', or 'scheduled'.".format(order_schedule))
            raise
        else:
            self.order_schedule = order_schedule

        self.start_num = start_num
        self.iter_num = iter_num if self.order_schedule == 'scheduled' else 1   # Number of rescheduled
        self.inner_iter_num = 10 if inner_iter_num is None else inner_iter_num  # Number of component-wise pgd updates
        self.step_size_pool = [2.5 * ((eps[1] - eps[0]) / 2) / self.inner_iter_num for eps in
                               self.eps_pool]  # 2.5 * ε-test / num_steps
        self.multiple_rand_start = multiple_rand_start  # False: start from little epsilon to the upper bound

        self.batch_size = self.adv_val_pool = self.eps_space = self.adv_val_space = self.curr_dsm = \
            self.curr_seq = self.is_attacked = self.is_not_attacked = None

    def _setup_attack(self):
        if self.multiple_rand_start:
            hue_space = torch.rand((self.start_num, self.batch_size), device=self.device) * (
                    self.eps_pool[0][1] - self.eps_pool[0][0]) + self.eps_pool[0][0]
            sat_space = torch.rand((self.start_num, self.batch_size), device=self.device) * (
                    self.eps_pool[1][1] - self.eps_pool[1][0]) + self.eps_pool[1][0]
            rot_space = torch.rand((self.start_num, self.batch_size), device=self.device) * (
                    self.eps_pool[2][1] - self.eps_pool[2][0]) + self.eps_pool[2][0]
            bright_space = torch.rand((self.start_num, self.batch_size), device=self.device) * (
                    self.eps_pool[3][1] - self.eps_pool[3][0]) + self.eps_pool[3][0]
            contrast_space = torch.rand((self.start_num, self.batch_size), device=self.device) * (
                    self.eps_pool[4][1] - self.eps_pool[4][0]) + self.eps_pool[4][0]
            linf_space = torch.tensor([self.eps_pool[5][1] for _ in range(self.start_num)], device=self.device)
        else:
            hue_space = torch.tensor([[((2 * i + 1.) / self.start_num) * (
                    self.eps_pool[0][1] - self.eps_pool[0][0]) / 2 + self.eps_pool[0][0] for _ in
                                       range(self.batch_size)] for i in range(self.start_num)], device=self.device)
            sat_space = torch.tensor([[((2 * i + 1.) / self.start_num) * (
                    self.eps_pool[1][1] - self.eps_pool[1][0]) / 2 + self.eps_pool[1][0] for _ in
                                       range(self.batch_size)] for i in range(self.start_num)], device=self.device)
            rot_space = torch.tensor([[((2 * i + 1.) / self.start_num) * (
                    self.eps_pool[2][1] - self.eps_pool[2][0]) / 2 + self.eps_pool[2][0] for _ in
                                       range(self.batch_size)] for i in range(self.start_num)], device=self.device)
            bright_space = torch.tensor([[((2 * i + 1.) / self.start_num) * (
                    self.eps_pool[3][1] - self.eps_pool[3][0]) / 2 + self.eps_pool[3][0] for _ in
                                          range(self.batch_size)] for i in range(self.start_num)], device=self.device)
            contrast_space = torch.tensor([[((2 * i + 1.) / self.start_num) * (
                    self.eps_pool[4][1] - self.eps_pool[4][0]) / 2 + self.eps_pool[4][0] for _ in
                                            range(self.batch_size)] for i in range(self.start_num)], device=self.device)
            linf_space = torch.tensor([self.eps_pool[5][1] for _ in range(self.start_num)], device=self.device)

        self.adv_val_pool = [hue_space, sat_space, rot_space, bright_space, contrast_space, linf_space]
        self.eps_space = [self.eps_pool[i] for i in self.enabled_attack]
        self.adv_val_space = [self.adv_val_pool[i] for i in self.enabled_attack]

        if self.curr_seq is None:
            self.curr_dsm = sinkhorn.initial_dsm(self.seq_num)
            self.curr_seq = sinkhorn.convert_dsm_to_sequence(self.curr_dsm)

            if self.order_schedule == 'fixed':
                self.fixed_order = tuple([self.enabled_attack.index(i) for i in self.fixed_order])
                self.curr_seq = torch.tensor(self.fixed_order).cuda(self.local_rank) if self.local_rank != -1 else \
                    torch.tensor(self.fixed_order).cuda()
                self.curr_dsm = sinkhorn.convert_seq_to_dsm(self.curr_seq)

            assert self.curr_seq is not None

    def forward(self, inputs, labels):
        if self.batch_size != inputs.shape[0]:
            self.batch_size = inputs.shape[0]
        self._setup_attack()

        if self.mode == 'eval':
            self.is_attacked = torch.zeros(self.batch_size).bool().cuda(
                self.local_rank) if self.local_rank != -1 else torch.zeros(self.batch_size).bool().cuda()
            self.is_not_attacked = torch.ones(self.batch_size).bool().cuda(
                self.local_rank) if self.local_rank != -1 else torch.ones(self.batch_size).bool().cuda()
            return self.attack_eval(inputs, labels)

        return self.attack_train(inputs, labels)

    def eval_hue(self, data, hue, labels):
        hue = hue.detach()
        hue[self.is_attacked] = 0
        hue.requires_grad_()
        ori_data = data.clone()
        ori_is_attacked = self.is_attacked

        new_data = kornia.enhance.adjust_hue(data, hue)
        for i in range(self.inner_iter_num):

            outputs = self.model(new_data)
            cur_pred = outputs.max(1, keepdim=True)[1].squeeze()
            # Get successfully attacked indexes
            self.is_attacked = torch.logical_or(ori_is_attacked, cur_pred != labels)

            self.model.zero_grad()
            cost = F.cross_entropy(outputs, labels)
            if i + 1 == self.inner_iter_num:
                hue_grad = torch.autograd.grad(cost, hue, retain_graph=True)[0]
            else:
                hue_grad = torch.autograd.grad(cost, hue, retain_graph=False)[0]
            hue_grad[self.is_attacked] = 0
            hue = torch.clamp(hue + torch.sign(hue_grad) * self.step_size_pool[0], self.eps_pool[0][0],
                              self.eps_pool[0][1]).detach().requires_grad_()
            new_data = kornia.enhance.adjust_hue(data, hue)

        new_data[ori_is_attacked] = ori_data[ori_is_attacked]

        return new_data, hue

    def eval_sat(self, data, sat, labels):
        sat = sat.detach()
        sat[self.is_attacked] = 1
        sat.requires_grad_()
        ori_data = data.clone()
        ori_is_attacked = self.is_attacked

        new_data = kornia.enhance.adjust_saturation(data, sat)
        for i in range(self.inner_iter_num):
            outputs = self.model(new_data)
            cur_pred = outputs.max(1, keepdim=True)[1].squeeze()
            # Get successfully attacked indexes
            self.is_attacked = torch.logical_or(ori_is_attacked, cur_pred != labels)

            self.model.zero_grad()
            cost = F.cross_entropy(outputs, labels)
            if i + 1 == self.inner_iter_num:
                sat_grad = torch.autograd.grad(cost, sat, retain_graph=True)[0]
            else:
                sat_grad = torch.autograd.grad(cost, sat, retain_graph=False)[0]
            sat_grad[self.is_attacked] = 0
            sat = torch.clamp(sat + torch.sign(sat_grad) * self.step_size_pool[1], self.eps_pool[1][0],
                              self.eps_pool[1][1]).detach().requires_grad_()
            new_data = kornia.enhance.adjust_saturation(data, sat)

        new_data[ori_is_attacked] = ori_data[ori_is_attacked]

        return new_data, sat

    def eval_rot(self, data, theta, labels):
        theta = theta.detach()
        theta[self.is_attacked] = 0
        theta.requires_grad_()
        ori_data = data.clone()
        ori_is_attacked = self.is_attacked

        new_data = kornia.geometry.transform.rotate(data, theta)
        for i in range(self.inner_iter_num):
            outputs = self.model(new_data)
            cur_pred = outputs.max(1, keepdim=True)[1].squeeze()
            # Get successfully attacked indexes
            self.is_attacked = torch.logical_or(ori_is_attacked, cur_pred != labels)

            self.model.zero_grad()
            cost = F.cross_entropy(outputs, labels)
            if i + 1 == self.inner_iter_num:
                theta_grad = torch.autograd.grad(cost, theta, retain_graph=True)[0]
            else:
                theta_grad = torch.autograd.grad(cost, theta, retain_graph=False)[0]
            theta_grad[self.is_attacked] = 0
            theta = torch.clamp(theta + torch.sign(theta_grad) * self.step_size_pool[2], self.eps_pool[2][0],
                                self.eps_pool[2][1]).detach().requires_grad_()
            new_data = kornia.geometry.transform.rotate(data, theta)

        new_data[ori_is_attacked] = ori_data[ori_is_attacked]

        return new_data, theta

    def eval_bright(self, data, brightness, labels):
        brightness = brightness.detach()
        brightness[self.is_attacked] = 0
        brightness.requires_grad_()
        ori_data = data.clone()
        ori_is_attacked = self.is_attacked

        new_data = kornia.enhance.adjust_brightness(data, brightness)
        for i in range(self.inner_iter_num):
            outputs = self.model(new_data)
            cur_pred = outputs.max(1, keepdim=True)[1].squeeze()
            # Get successfully attacked indexes
            self.is_attacked = torch.logical_or(ori_is_attacked, cur_pred != labels)

            self.model.zero_grad()
            cost = F.cross_entropy(outputs, labels)
            if i + 1 == self.inner_iter_num:
                brightness_grad = torch.autograd.grad(cost, brightness, retain_graph=True)[0]
            else:
                brightness_grad = torch.autograd.grad(cost, brightness, retain_graph=False)[0]
            brightness_grad[self.is_attacked] = 0
            brightness = torch.clamp(brightness + torch.sign(brightness_grad) * self.step_size_pool[3],
                                     self.eps_pool[3][0],
                                     self.eps_pool[3][1]).detach().requires_grad_()
            new_data = kornia.enhance.adjust_brightness(data, brightness)

        new_data[ori_is_attacked] = ori_data[ori_is_attacked]

        return new_data, brightness

    def eval_contrast(self, data, contrast, labels):
        contrast = contrast.detach()
        contrast[self.is_attacked] = 1
        contrast.requires_grad_()
        ori_data = data.clone()
        ori_is_attacked = self.is_attacked

        new_data = kornia.enhance.adjust_contrast(data, contrast)
        for i in range(self.inner_iter_num):

            outputs = self.model(new_data)
            cur_pred = outputs.max(1, keepdim=True)[1].squeeze()
            # Get successfully attacked indexes
            self.is_attacked = torch.logical_or(ori_is_attacked, cur_pred != labels)

            self.model.zero_grad()
            cost = F.cross_entropy(outputs, labels)
            if i + 1 == self.inner_iter_num:
                contrast_grad = torch.autograd.grad(cost, contrast, retain_graph=True)[0]
            else:
                contrast_grad = torch.autograd.grad(cost, contrast, retain_graph=False)[0]
            contrast_grad[self.is_attacked] = 0
            contrast = torch.clamp(contrast + torch.sign(contrast_grad) * self.step_size_pool[4], self.eps_pool[4][0],
                                   self.eps_pool[4][1]).detach().requires_grad_()
            new_data = kornia.enhance.adjust_contrast(data, contrast)

        new_data[ori_is_attacked] = ori_data[ori_is_attacked]

        return new_data, contrast

    def eval_linf(self, data, labels):
        ori_data = data.detach()
        data.detach_().requires_grad_()
        ori_is_attacked = self.is_attacked

        for i in range(self.inner_iter_num):
            outputs = self.model(data)
            cur_pred = outputs.max(1, keepdim=True)[1].squeeze()
            # Get successfully attacked indexes
            self.is_attacked = torch.logical_or(ori_is_attacked, cur_pred != labels)

            self.model.zero_grad()
            cost = F.cross_entropy(outputs, labels)
            if i + 1 == self.inner_iter_num:
                img_grad = torch.autograd.grad(cost, data, retain_graph=True)[0]
            else:
                img_grad = torch.autograd.grad(cost, data, retain_graph=False)[0]
            img_grad[self.is_attacked] = 0
            adv_data = data + self.step_size_pool[5] * torch.sign(img_grad)
            eta = torch.clamp(adv_data - ori_data, min=self.eps_pool[5][0], max=self.eps_pool[5][1])
            data = torch.clamp(ori_data + eta, min=0., max=1.).detach_().requires_grad_()

        return data

    def train_hue(self, data, hue, labels):
        hue = hue.detach().requires_grad_()

        new_data = kornia.enhance.adjust_hue(data, hue)
        if self.mode == 'fast_train':
            self.model.zero_grad()
            return new_data, hue

        for i in range(self.inner_iter_num):
            outputs = self.model(new_data)
            self.model.zero_grad()
            cost = F.cross_entropy(outputs, labels)
            if i + 1 == self.inner_iter_num:
                hue_grad = torch.autograd.grad(cost, hue, retain_graph=True)[0]
            else:
                hue_grad = torch.autograd.grad(cost, hue, retain_graph=False)[0]
            hue = torch.clamp(hue + torch.sign(hue_grad) * self.step_size_pool[0], self.eps_pool[0][0],
                              self.eps_pool[0][1]).detach().requires_grad_()
            new_data = kornia.enhance.adjust_hue(data, hue)

        return new_data, hue

    def train_sat(self, data, sat, labels):
        sat = sat.detach().requires_grad_()

        new_data = kornia.enhance.adjust_saturation(data, sat)
        if self.mode == 'fast_train':
            self.model.zero_grad()
            return new_data, sat

        for i in range(self.inner_iter_num):
            outputs = self.model(new_data)
            self.model.zero_grad()
            cost = F.cross_entropy(outputs, labels)
            if i + 1 == self.inner_iter_num:
                sat_grad = torch.autograd.grad(cost, sat, retain_graph=True)[0]
            else:
                sat_grad = torch.autograd.grad(cost, sat, retain_graph=False)[0]
            sat = torch.clamp(sat + torch.sign(sat_grad) * self.step_size_pool[1], self.eps_pool[1][0],
                              self.eps_pool[1][1]).detach().requires_grad_()
            new_data = kornia.enhance.adjust_saturation(data, sat)

        return new_data, sat

    def train_rot(self, data, theta, labels):
        theta = theta.detach().requires_grad_()

        new_data = kornia.geometry.transform.rotate(data, theta)
        if self.mode == 'fast_train':
            self.model.zero_grad()
            return new_data, theta

        for i in range(self.inner_iter_num):
            outputs = self.model(new_data)
            self.model.zero_grad()
            cost = F.cross_entropy(outputs, labels)
            if i + 1 == self.inner_iter_num:
                theta_grad = torch.autograd.grad(cost, theta, retain_graph=True)[0]
            else:
                theta_grad = torch.autograd.grad(cost, theta, retain_graph=False)[0]
            theta = torch.clamp(theta + torch.sign(theta_grad) * self.step_size_pool[2], self.eps_pool[2][0],
                                self.eps_pool[2][1]).detach().requires_grad_()
            new_data = kornia.geometry.transform.rotate(data, theta)

        return new_data, theta

    def train_bright(self, data, brightness, labels):
        brightness = brightness.detach().requires_grad_()

        new_data = kornia.enhance.adjust_brightness(data, brightness)
        if self.mode == 'fast_train':
            self.model.zero_grad()
            return new_data, brightness

        for i in range(self.inner_iter_num):
            outputs = self.model(new_data)
            self.model.zero_grad()
            cost = F.cross_entropy(outputs, labels)
            if i + 1 == self.inner_iter_num:
                brightness_grad = torch.autograd.grad(cost, brightness, retain_graph=True)[0]
            else:
                brightness_grad = torch.autograd.grad(cost, brightness, retain_graph=False)[0]
            brightness = torch.clamp(brightness + torch.sign(brightness_grad) * self.step_size_pool[3],
                                     self.eps_pool[3][0],
                                     self.eps_pool[3][1]).detach().requires_grad_()
            new_data = kornia.enhance.adjust_brightness(data, brightness)

        return new_data, brightness

    def train_contrast(self, data, contrast, labels):
        contrast = contrast.detach().requires_grad_()

        new_data = kornia.enhance.adjust_contrast(data, contrast)
        if self.mode == 'fast_train':
            self.model.zero_grad()
            return new_data, contrast

        for i in range(self.inner_iter_num):
            outputs = self.model(new_data)
            self.model.zero_grad()
            cost = F.cross_entropy(outputs, labels)
            if i + 1 == self.inner_iter_num:
                contrast_grad = torch.autograd.grad(cost, contrast, retain_graph=True)[0]
            else:
                contrast_grad = torch.autograd.grad(cost, contrast, retain_graph=False)[0]
            contrast = torch.clamp(contrast + torch.sign(contrast_grad) * self.step_size_pool[4], self.eps_pool[4][0],
                                   self.eps_pool[4][1]).detach().requires_grad_()
            new_data = kornia.enhance.adjust_contrast(data, contrast)

        return new_data, contrast

    def train_linf(self, data, labels):
        ori_data = data.detach()

        for i in range(self.inner_iter_num):
            outputs = self.model(data)
            self.model.zero_grad()
            cost = F.cross_entropy(outputs, labels)
            if i + 1 == self.inner_iter_num:
                img_grad = torch.autograd.grad(cost, data, retain_graph=True)[0]
            else:
                img_grad = torch.autograd.grad(cost, data, retain_graph=False)[0]
            adv_data = data + self.step_size_pool[5] * torch.sign(img_grad)

            eta = torch.clamp(adv_data - ori_data, min=self.eps_pool[5][0], max=self.eps_pool[5][1])
            data = torch.clamp(ori_data + eta, min=0., max=1.).detach().requires_grad_()

        return data

    def update_dsm(self, adv_img, labels):
        """
        Updating order
        """

        if self.order_schedule == 'fixed':
            pass
        elif self.order_schedule == 'random':
            self.curr_dsm = sinkhorn.initial_dsm(self.seq_num)
            self.curr_seq = sinkhorn.convert_dsm_to_sequence(self.curr_dsm)
        elif self.order_schedule == 'scheduled':

            outputs = self.model(adv_img)
            self.model.zero_grad()

            cost = F.cross_entropy(outputs, labels)
            cost.backward(retain_graph=True)
            dsm_grad = torch.autograd.grad(cost, self.curr_dsm, create_graph=False, retain_graph=False)[0]
            half_dsm = self.curr_dsm * torch.exp(-dsm_grad * 0.5)
            self.curr_dsm = sinkhorn.my_sinkhorn(half_dsm.detach()).requires_grad_()
            self.curr_dsm = torch.squeeze(self.curr_dsm, 0)
            self.curr_seq = sinkhorn.convert_dsm_to_sequence(self.curr_dsm)

            # if repeat, do hungarian
            while sinkhorn.is_repeated(self.curr_seq):
                self.curr_seq = sinkhorn.my_matching(self.curr_dsm.detach())
                self.curr_dsm = sinkhorn.convert_seq_to_dsm(self.curr_seq).requires_grad_()

    def attack_eval(self, images, labels):
        attack = self.attack_dict
        attack_num = len(self.curr_seq)
        adv_img = images
        if self.local_rank != -1:
            adv_val_saved = torch.zeros((attack_num, self.batch_size)).cuda(self.local_rank)
        else:
            adv_val_saved = torch.zeros((attack_num, self.batch_size)).cuda()
        for i in range(self.start_num):
            adv_val = [self.adv_val_space[idx][i] for idx in range(attack_num)]
            if self.is_attacked.sum() > 0:
                for att_id in range(attack_num):
                    if att_id == self.linf_idx:
                        continue
                    adv_val[att_id].detach()
                    adv_val[att_id][self.is_attacked] = adv_val_saved[att_id][self.is_attacked]
                    adv_val[att_id].requires_grad_()

            if self.order_schedule == 'scheduled':
                self.curr_dsm.requires_grad_()
            else:
                self.update_dsm(adv_img, labels)
            for j in range(self.iter_num):
                # print('iter : {} order update --> {}'.format(j, self.curr_seq.detach().cpu()))
                self.is_not_attacked = torch.logical_not(self.is_attacked)
                adv_img = adv_img.detach()
                adv_img[self.is_not_attacked] = images.data[self.is_not_attacked]
                adv_img.requires_grad = True

                for tdx in range(attack_num):
                    idx = self.curr_seq[tdx]

                    if self.order_schedule == 'scheduled':
                        m = self.curr_dsm[tdx][idx]
                        if idx == self.linf_idx:
                            adv_img = (self.curr_dsm[tdx][idx] / m) * attack[idx](adv_img, labels)
                        else:
                            adv_img, adv_val_updated = attack[idx](adv_img, adv_val[idx], labels)
                            adv_img = (self.curr_dsm[tdx][idx] / m) * adv_img
                            adv_val[idx] = adv_val_updated

                    else:
                        if idx == self.linf_idx:
                            adv_img = attack[idx](adv_img, labels)
                        else:
                            adv_img, adv_val_updated = attack[idx](adv_img, adv_val[idx], labels)
                            adv_val[idx] = adv_val_updated

                    for att_id in range(attack_num):
                        if att_id == self.linf_idx:
                            continue
                        adv_val_saved[att_id][self.is_attacked] = adv_val[att_id][self.is_attacked].detach()

                    if self.is_attacked.sum() == self.batch_size:
                        break

                if self.is_attacked.sum() == self.batch_size:
                    break
                elif 1 < self.iter_num != j:
                    self.update_dsm(adv_img, labels)

        return adv_img

    def attack_train(self, images, labels):
        attack = self.attack_dict
        attack_num = len(self.curr_seq)
        adv_img = images

        for i in range(self.start_num):
            adv_val = [self.adv_val_space[idx][i].requires_grad_() for idx in range(attack_num)]

            if self.order_schedule == 'scheduled':
                self.curr_dsm.requires_grad_()
            else:
                self.update_dsm(adv_img, labels)
            for j in range(self.iter_num):
                adv_img = adv_img.detach()
                adv_img.requires_grad = True

                for tdx in range(attack_num):
                    idx = self.curr_seq[tdx]

                    if self.order_schedule == 'scheduled':
                        m = self.curr_dsm[tdx][idx]
                        if idx == self.linf_idx:
                            adv_img = (self.curr_dsm[tdx][idx] / m) * attack[idx](adv_img, labels)
                        else:
                            adv_img, adv_val[idx] = attack[idx](adv_img, adv_val[idx], labels)
                            adv_img = (self.curr_dsm[tdx][idx] / m) * adv_img
                    else:
                        if idx == self.linf_idx:
                            adv_img = attack[idx](adv_img, labels)
                        else:
                            adv_img, adv_val[idx] = attack[idx](adv_img, adv_val[idx], labels)

                if 1 < self.iter_num != j:
                    self.update_dsm(adv_img, labels)

            adv_img = adv_img.detach()

        return adv_img
