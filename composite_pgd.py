import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from math import pi

import warnings
import sinkhorn_ops as sinkhorn

warnings.filterwarnings('ignore')


class PGD(nn.Module):
    def __init__(self, model, enabled_attack,
                 hue_epsilon=(-pi, pi), sat_epsilon=(0.7, 2), rot_epsilon=(-10, 10),
                 bright_epsilon=(0.2, 0.2), contrast_epsilon=(0.7, 1.5), linf_epsilon=8 / 255,
                 start_num=10, iter_num=10, inner_iter_num=10, multiple_rand_start=True, order_schedule='fixed'):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss().cuda()
        self.loss_kl = nn.KLDivLoss(size_average=False).cuda()
        self.fixed_order = enabled_attack
        self.enabled_attack = tuple(sorted(enabled_attack))
        self.seq_num = len(enabled_attack)  # attack_num
        self.attack_pool = (self.new_hue, self.new_sat, self.new_rot, self.new_bright, self.new_contrast, self.l_inf)
        self.attack_dict = tuple([self.attack_pool[i] for i in self.enabled_attack])
        self.linf_idx = self.enabled_attack.index(5) if 5 in self.enabled_attack else None

        self.eps_pool = torch.tensor(
            [hue_epsilon, sat_epsilon, rot_epsilon, bright_epsilon, contrast_epsilon, linf_epsilon]).cuda()
        self.start_num = start_num
        self.iter_num = iter_num
        self.inner_iter_num = int(5/self.start_num) if self.start_num != 0 else 10 if inner_iter_num is None else inner_iter_num
        self.step_size_pool = [0.05 * (eps[1] - eps[0]) for eps in self.eps_pool]
        self.multiple_rand_start = multiple_rand_start  # False: start from little epsilon to the upper bound
        self.order_schedule = order_schedule

        self.batch_size = self.adv_val_pool = self.eps_space = self.adv_val_space = self.curr_dsm = \
            self.curr_seq = self.is_attacked = self.is_not_attacked = None

    def _setup_attack(self):
        if self.multiple_rand_start:
            hue_space = torch.rand((self.start_num, self.batch_size), device='cuda') * (
                    self.eps_pool[0][1] - self.eps_pool[0][0]) + self.eps_pool[0][0]
            sat_space = torch.rand((self.start_num, self.batch_size), device='cuda') * (
                    self.eps_pool[1][1] - self.eps_pool[1][0]) + self.eps_pool[1][0]
            rot_space = torch.rand((self.start_num, self.batch_size), device='cuda') * (
                    self.eps_pool[2][1] - self.eps_pool[2][0]) + self.eps_pool[2][0]
            bright_space = torch.rand((self.start_num, self.batch_size), device='cuda') * (
                    self.eps_pool[3][1] - self.eps_pool[3][0]) + self.eps_pool[3][0]
            contrast_space = torch.rand((self.start_num, self.batch_size), device='cuda') * (
                    self.eps_pool[4][1] - self.eps_pool[4][0]) + self.eps_pool[4][0]
            linf_space = torch.tensor([self.eps_pool[5][1] for _ in range(self.start_num)], device='cuda')
        else:
            hue_space = torch.tensor([[((2 * i + 1.) / self.start_num) * (
                        self.eps_pool[0][1] - self.eps_pool[0][0]) / 2 + self.eps_pool[0][0] for _ in
                                       range(self.batch_size)] for i in range(self.start_num)], device='cuda')
            sat_space = torch.tensor([[((2 * i + 1.) / self.start_num) * (
                        self.eps_pool[1][1] - self.eps_pool[1][0]) / 2 + self.eps_pool[1][0] for _ in
                                       range(self.batch_size)] for i in range(self.start_num)], device='cuda')
            rot_space = torch.tensor([[((2 * i + 1.) / self.start_num) * (
                        self.eps_pool[2][1] - self.eps_pool[2][0]) / 2 + self.eps_pool[2][0] for _ in
                                       range(self.batch_size)] for i in range(self.start_num)], device='cuda')
            bright_space = torch.tensor([[((2 * i + 1.) / self.start_num) * (
                        self.eps_pool[3][1] - self.eps_pool[3][0]) / 2 + self.eps_pool[3][0] for _ in
                                          range(self.batch_size)] for i in range(self.start_num)], device='cuda')
            contrast_space = torch.tensor([[((2 * i + 1.) / self.start_num) * (
                        self.eps_pool[4][1] - self.eps_pool[4][0]) / 2 + self.eps_pool[4][0] for _ in
                                            range(self.batch_size)] for i in range(self.start_num)], device='cuda')
            linf_space = torch.tensor([self.eps_pool[5][1] for _ in range(self.start_num)], device='cuda')

        self.adv_val_pool = [hue_space, sat_space, rot_space, bright_space, contrast_space, linf_space]
        self.eps_space = [self.eps_pool[i] for i in self.enabled_attack]
        self.adv_val_space = [self.adv_val_pool[i] for i in self.enabled_attack]

        self.curr_dsm = sinkhorn.initial_dsm(self.seq_num)
        self.curr_seq = sinkhorn.convert_dsm_to_sequence(self.curr_dsm)
        if self.order_schedule == 'fixed':
            self.fixed_order = tuple([self.enabled_attack.index(i) for i in self.fixed_order])
            self.curr_seq = torch.tensor(self.fixed_order).cuda()
            self.curr_dsm = sinkhorn.convert_seq_to_dsm(self.curr_seq)

    def forward(self, inputs, labels):
        self.batch_size = inputs.shape[0]
        if self.curr_seq is None:
            self._setup_attack()
        assert self.curr_seq is not None

        self.is_attacked = torch.zeros(self.batch_size).bool().cuda()
        self.is_not_attacked = torch.ones(self.batch_size).bool().cuda()
        return self.do_attack_together(inputs, labels)[0]

    def rgb_to_hsv(self, arr):
        assert (arr.shape[0] == 3)
        t_arr = arr.permute(1, 2, 0)
        r = t_arr[..., 0]
        g = t_arr[..., 1]
        b = t_arr[..., 2]
        in_shape = t_arr.shape
        out = torch.zeros_like(t_arr)
        arr_max = torch.max(t_arr, -1)[0]
        arr_min = torch.min(t_arr, -1)[0]
        ipos = arr_max > 0
        delta = (arr_max - arr_min)
        z = torch.zeros(1).cuda()
        s = torch.zeros_like(delta)  # saturation
        s[ipos] = delta[ipos] / arr_max[ipos]
        ipos = delta > 0

        # red is max
        idx = (r == arr_max) & ipos
        out[..., 0][idx] = (g[idx] - b[idx]) / delta[idx]
        # green is max
        idx = (g == arr_max) & ipos
        out[..., 0][idx] = (b[idx] - r[idx]) / delta[idx] + 2.
        # blue is max
        idx = (b == arr_max) & ipos
        out[..., 0][idx] = (r[idx] - g[idx]) / delta[idx] + 4.

        out[..., 0] = (out[..., 0] / 6.0) % 1.0
        out[..., 1] = s
        out[..., 2] = arr_max

        return out.reshape(in_shape)

    def hsv_to_rgb(self, hsv):
        in_shape = hsv.shape
        out = torch.zeros_like(hsv)

        h = hsv[..., 0]
        s = hsv[..., 1]
        v = hsv[..., 2]

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        i = torch.tensor(h * 6.0, dtype=torch.int8).cuda()
        f = (h * 6.0 - i)
        p = (v * (1.0 - s))
        q = (v * (1.0 - s * f))
        t = (v * (1.0 - s * (1.0 - f)))
        z = torch.zeros(1).cuda()

        idx = (i % 6 == 0)
        r[idx] = v[idx]
        g[idx] = t[idx]
        b[idx] = p[idx]

        idx = (i % 6 == 1)
        r[idx] = q[idx]
        g[idx] = v[idx]
        b[idx] = p[idx]

        idx = (i % 6 == 2)
        r[idx] = p[idx]
        g[idx] = v[idx]
        b[idx] = t[idx]

        idx = (i % 6 == 3)
        r[idx] = p[idx]
        g[idx] = q[idx]
        b[idx] = v[idx]

        idx = (i % 6 == 4)
        r[idx] = t[idx]
        g[idx] = p[idx]
        b[idx] = v[idx]

        idx = (i % 6 == 5)
        r[idx] = v[idx]
        g[idx] = p[idx]
        b[idx] = q[idx]

        idx = (s == 0)
        r[idx] = v[idx]
        g[idx] = v[idx]
        b[idx] = v[idx]

        out[..., 0] = r
        out[..., 1] = g
        out[..., 2] = b
        return out.reshape(in_shape)

    def new_hue(self, data, hue, labels):
        # assert data != None
        # new_data = torch.tensor([]).cuda()

        # for image in data:
        #     #rgb to hsv
        #     #data_hsv = self.rgb_to_hsv(self.un_normalize(image))
        #     data_hsv = self.rgb_to_hsv(image)
        #     data_hsv[:, :, 0] = (data_hsv[:, :, 0] + adv_hue) % 1.0

        #     #hsv to rgb
        #     #data_hsv2rgb = self.normalize(self.hsv_to_rgb(data_hsv))
        #     data_hsv2rgb = self.hsv_to_rgb(data_hsv)
        #     new_data = torch.cat((new_data, data_hsv2rgb.permute(2,0,1).unsqueeze(0)))

        hue.detach().requires_grad_()
        new_data = torch.clone(data)

        for i in range(self.inner_iter_num):
            # for image in data:
            #     # rgb to hsv
            #     data_hsv = self.rgb_to_hsv(image)
            #     # data_hsv = self.rgb_to_hsv(image)
            #     data_hsv[:, :, 0] = (data_hsv[:, :, 0] + hue) % 1.0
            #
            #     # hsv to rgb
            #     data_hsv2rgb = self.hsv_to_rgb(data_hsv)
            #     # data_hsv2rgb = self.hsv_to_rgb(data_hsv)
            #     new_data = torch.cat((new_data, data_hsv2rgb.permute(2, 0, 1).unsqueeze(0)))
            # data = new_data
            new_data = kornia.adjust_hue(data, hue)

            outputs = self.model(new_data)
            self.model.zero_grad()
            cost = self.loss(outputs, labels)
            if i + 1 == self.inner_iter_num:
                hue_grad = torch.autograd.grad(cost, hue, retain_graph=True)[0]
            else:
                hue_grad = torch.autograd.grad(cost, hue, retain_graph=False)[0]
            hue = torch.clamp(hue + torch.sign(hue_grad) * self.step_size_pool[0], self.eps_pool[0][0],
                              self.eps_pool[0][1]).detach().requires_grad_()

        return new_data, hue

    def new_sat(self, data, sat, labels):
        # assert data != None
        # new_data = torch.tensor([]).cuda()

        # for image in data:
        #     #rgb to hsv
        #     #data_hsv = self.rgb_to_hsv(un_normalize(image))
        #     data_hsv = self.rgb_to_hsv(image)
        #     data_hsv[:, :, 1] = torch.clamp(data_hsv[:, :, 1] + adv_sat, 0., 1.)

        #     #hsv to rgb
        #     #data_hsv2rgb = normalize(self.hsv_to_rgb(data_hsv))
        #     data_hsv2rgb = self.hsv_to_rgb(data_hsv)
        #     new_data = torch.cat((new_data, data_hsv2rgb.permute(2,0,1).unsqueeze(0)))

        sat.detach().requires_grad_()
        new_data = torch.clone(data)

        for i in range(self.inner_iter_num):
            # for image in data:
            #     # rgb to hsv
            #     data_hsv = self.rgb_to_hsv(image)
            #     data_hsv[:, :, 1] = torch.clamp(data_hsv[:, :, 1] + sat, 0., 1.)
            #
            #     # hsv to rgb
            #     data_hsv2rgb = self.hsv_to_rgb(data_hsv)
            #     new_data = torch.cat((new_data, data_hsv2rgb.permute(2, 0, 1).unsqueeze(0)))
            # data = new_data
            new_data = kornia.adjust_saturation(data, sat)

            outputs = self.model(new_data)
            cur_pred = outputs.max(1, keepdim=True)[1]
            # if cur_pred.item() != labels.item():
            #     self.is_attacked = True
            #     break
            self.model.zero_grad()
            cost = self.loss(outputs, labels)
            if i + 1 == self.inner_iter_num:
                sat_grad = torch.autograd.grad(cost, sat, retain_graph=True)[0]
            else:
                sat_grad = torch.autograd.grad(cost, sat, retain_graph=False)[0]
            sat = torch.clamp(sat + torch.sign(sat_grad) * self.step_size_pool[1], self.eps_pool[1][0],
                              self.eps_pool[1][1]).detach().requires_grad_()

        return new_data, sat

    def new_rot(self, data, theta, labels):
        theta.detach().requires_grad_()
        new_data = torch.clone(data)

        for i in range(self.inner_iter_num):
            angle = torch.ones(data.shape[0]).cuda() * theta

            center = torch.ones(data.shape[0], 2).cuda()
            center[..., 0] = data.shape[2] / 2  # x
            center[..., 1] = data.shape[3] / 2  # y
            scale = torch.ones(data.shape[0], 2).cuda()
            M = kornia.get_rotation_matrix2d(center, angle, scale)
            new_data = kornia.warp_affine(data, M, dsize=(data.shape[2], data.shape[3]))

            outputs = self.model(new_data)
            cur_pred = outputs.max(1, keepdim=True)[1]
            # if cur_pred.item() != labels.item():
            #     self.is_attacked = True
            #     break
            self.model.zero_grad()
            cost = self.loss(outputs, labels).cuda()
            if i + 1 == self.inner_iter_num:
                theta_grad = torch.autograd.grad(cost, theta, retain_graph=True)[0]
            else:
                theta_grad = torch.autograd.grad(cost, theta, retain_graph=False)[0]
            theta = torch.clamp(theta + torch.sign(theta_grad) * self.step_size_pool[2], self.eps_pool[2][0],
                                self.eps_pool[2][1]).detach().requires_grad_()
        return new_data, theta

    def new_bright(self, data, brightness, labels):
        brightness.detach().requires_grad_()
        new_data = torch.clone(data)

        for i in range(self.inner_iter_num):
            # new_data = data + brightness  # We restrict that the images are in [0,1]
            # new_data = torch.min(torch.max(new_data, torch.zeros_like(new_data)),
            #                      torch.ones_like(new_data))

            new_data = kornia.adjust_brightness(data, brightness)

            outputs = self.model(new_data)
            cur_pred = outputs.max(1, keepdim=True)[1]
            # if cur_pred.item() != labels.item():
            #     self.is_attacked = True
            #     break
            self.model.zero_grad()
            cost = self.loss(outputs, labels)
            if i + 1 == self.inner_iter_num:
                brightness_grad = torch.autograd.grad(cost, brightness, retain_graph=True)[0]
            else:
                brightness_grad = torch.autograd.grad(cost, brightness, retain_graph=False)[0]
            brightness = torch.clamp(brightness + torch.sign(brightness_grad) * self.step_size_pool[3],
                                     self.eps_pool[3][0],
                                     self.eps_pool[3][1]).detach().requires_grad_()

        return new_data, brightness

    def new_contrast(self, data, contrast, labels):
        contrast.detach().requires_grad_()
        new_data = torch.clone(data)

        for i in range(self.inner_iter_num):
            # new_data = data*((1+contrast).view(-1, 1, 1, 1))
            # new_data = torch.min(torch.max(new_data, torch.zeros_like(new_data)),
            #                      torch.ones_like(new_data))

            new_data = kornia.adjust_contrast(data, contrast)

            outputs = self.model(new_data)
            cur_pred = outputs.max(1, keepdim=True)[1]
            # if cur_pred.item() != labels.item():
            #     self.is_attacked = True
            #     break
            self.model.zero_grad()
            cost = self.loss(outputs, labels)
            if i + 1 == self.inner_iter_num:
                contrast_grad = torch.autograd.grad(cost, contrast, retain_graph=True)[0]
            else:
                contrast_grad = torch.autograd.grad(cost, contrast, retain_graph=False)[0]
            contrast = torch.clamp(contrast + torch.sign(contrast_grad) * self.step_size_pool[4], self.eps_pool[4][0],
                                   self.eps_pool[4][1]).detach().requires_grad_()

        return new_data, contrast

    def l_inf(self, data, labels):
        ori_data = data.detach()

        for _ in range(self.inner_iter_num):
            data.requires_grad_()
            with torch.enable_grad():
                loss_kl = self.loss_kl(F.log_softmax(self.model(data), dim=1),
                                       F.softmax(self.model(ori_data), dim=1))
            grad = torch.autograd.grad(loss_kl, [data])[0]
            data = data.detach() + self.step_size_pool[5] * torch.sign(grad.detach())
            data = torch.min(torch.max(data, ori_data - self.eps_pool[5][1]), ori_data + self.eps_pool[5][1])
            data = torch.clamp(data, 0.0, 1.0)

        # for i in range(self.inner_iter_num):
        #     outputs = self.model(data)
        #     cur_pred = outputs.max(1, keepdim=True)[1]
        #     # if cur_pred.item() != labels.item():
        #     #     self.is_attacked = True
        #     #     break
        #
        #     self.model.zero_grad()
        #     cost = self.loss(outputs, labels) # change loss
        #     # cost.backward()
        #     # adv_data = data + self.step_size_pool[5] * data.grad.sign()
        #
        #     if i + 1 == self.inner_iter_num:
        #         img_grad = torch.autograd.grad(cost, data, retain_graph=True)[0]
        #     else:
        #         img_grad = torch.autograd.grad(cost, data, retain_graph=False)[0]
        #     adv_data = data + self.step_size_pool[5] * torch.sign(img_grad)
        #
        #     eta = torch.clamp(adv_data - ori_data, min=self.eps_pool[5][0], max=self.eps_pool[5][1])
        #     data = torch.clamp(ori_data + eta, min=0., max=1.).detach_().requires_grad_()

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

            cost = self.loss(outputs, labels)

            dsm_grad = torch.autograd.grad(cost, self.curr_dsm, create_graph=False, retain_graph=False)[0]

            half_dsm = self.curr_dsm * torch.exp(-dsm_grad * 0.2)
            self.curr_dsm = sinkhorn.my_sinkhorn(half_dsm.detach_()).requires_grad_()
            self.curr_seq = sinkhorn.convert_dsm_to_sequence(self.curr_dsm)

            # if repeat, do hungarian
            while sinkhorn.is_repeated(self.curr_seq):
                self.curr_seq = sinkhorn.my_matching(self.curr_dsm)
                self.curr_dsm = sinkhorn.convert_seq_to_dsm(self.curr_seq).requires_grad_()

    def do_attack_together(self, images, labels):
        attack = self.attack_dict
        attack_num = len(self.curr_seq)
        adv_img = None

        for i in range(self.start_num):
            adv_val = [self.adv_val_space[idx][i].requires_grad_() for idx in range(attack_num)]

            if self.order_schedule == 'scheduled':
                self.curr_dsm.requires_grad_()
            for j in range(self.iter_num):
                adv_img = images.data
                adv_img.requires_grad = True

                for tdx in range(attack_num):
                    idx = self.curr_seq[tdx]

                    if self.order_schedule == 'scheduled':
                        m = self.curr_dsm[tdx][idx]
                        if idx == self.linf_idx:
                            adv_img = (self.curr_dsm[tdx][idx] / m) * attack[idx](adv_img, labels)
                        else:
                            adv_obj = attack[idx](adv_img, adv_val[idx], labels)
                            adv_img = (self.curr_dsm[tdx][idx] / m) * adv_obj[0]
                            adv_val[idx] = adv_obj[1]
                    else:
                        if idx == self.linf_idx:
                            adv_img = attack[idx](adv_img, labels)
                        else:
                            adv_obj = attack[idx](adv_img, adv_val[idx], labels)
                            adv_img = adv_obj[0]
                            adv_val[idx] = adv_obj[1]

                    self.update_dsm(adv_img, labels)

            images = adv_img.detach()

        return images, self.curr_seq.detach()

    def cal_loss_landscape(self, images, labels, points):
        images = images.cuda()
        labels = labels.cuda()
        loss = nn.CrossEntropyLoss()
        enableSat = None

        x_axis = torch.tensor([(1. / points) * i for i in range(points)]).cuda()
        y_axis = torch.tensor([(1. / points) * i for i in range(points)]).cuda() if enableSat else torch.zeros(
            1)
        z_axis = torch.tensor([]).cuda()
        predict = torch.tensor([]).cuda()
        predict_correct = torch.zeros(1).cuda()
        predict_wrong = torch.ones(1).cuda()
        for hue_v in x_axis:
            for sat_v in y_axis:
                adv_img = self.new_hue(images, hue_v, sat_v)
                outputs = self.model(adv_img)

                z_axis = torch.cat((z_axis, loss(outputs, labels).unsqueeze(0)))
                cur_pred = outputs.max(1, keepdim=True)[1]
                if cur_pred.item() != labels.item():
                    predict = torch.cat((predict, predict_wrong))
                else:
                    predict = torch.cat((predict, predict_correct))
        if enableSat:
            return x_axis, y_axis, z_axis, predict
        else:
            return x_axis, z_axis, predict
