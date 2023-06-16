# -*- coding: utf-8 -*-

"""
Created on 03/23/2022
pruning.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import random

import copy
import types
from tqdm import tqdm


# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L11
def fetch_data(dataloader, num_classes, samples_per_class, mode=0, dm=0):
    if mode == 0:
        datas = [[] for _ in range(num_classes)]
        labels = [[] for _ in range(num_classes)]
        mark = dict()
        dataloader_iter = iter(dataloader)
        while True:
            inputs, targets = next(dataloader_iter)
            for idx in range(inputs.shape[0]):
                x, y = inputs[idx:idx + 1], targets[idx:idx + 1]
                category = y.item()
                if len(datas[category]) == samples_per_class:
                    mark[category] = True
                    continue
                datas[category].append(x)
                labels[category].append(y)
            if len(mark) == num_classes:
                break

        X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)

        if dm == 1:  # different label groups
            _index = []
            for i in range(samples_per_class):
                _index.extend([i + j * samples_per_class for j in range(0, num_classes)])
            X = X[_index]
            y = y[_index]
        elif dm == 2:  # different label groups and random
            _index = []
            _random_group = list(range(0, num_classes))
            random.shuffle(_random_group)
            _str_group = ''
            for i, g in enumerate(_random_group):
                if i % 2 == 0: _str_group += f'[{g},'
                else: _str_group += f'{g}] '
            print(_str_group)
            for i in _random_group:
                _index.extend([i * samples_per_class + j for j in range(0, samples_per_class)])
            X = X[_index]
            y = y[_index]

    else:
        dataloader_iter = iter(dataloader)
        inputs, targets = next(dataloader_iter)
        X, y = inputs[0:samples_per_class * num_classes], targets[0:samples_per_class * num_classes]
    return X, y


# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49
def hessian_gradient_product(net, samples, device, config, T=200, reinit=False):
    """
        data_mode:
            0 - 不同标签分组
            1 - 同标签分组
            2 - 随机不同标签组
            9 - 完全随机
        gard_mode:
            0 - 梯度范数梯度
            1 - 不同组点积（相似度）梯度
            2 - 不同组对应点积
            3 - 敏感于损失
    """

    data_mode = config.data_mode
    grad_mode = config.grad_mode
    num_group = config.num_group
    samples_per_class = config.samples_per_class
    num_classes = config.classe

    weights = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if isinstance(layer, nn.Linear) and reinit:
                nn.init.xavier_normal_(layer.weight)
            weights.append(layer.weight)
    for w in weights:
        w.requires_grad_(True)

    # print("fetch data")
    if data_mode == 9:
        sample_mode = 1
    else:
        sample_mode = 0
        if samples_per_class == 5 and num_group == 2:
            samples_per_class = 6
    inputs, targets = samples if isinstance(samples, tuple) else fetch_data(samples, num_classes, samples_per_class, mode=sample_mode, dm=data_mode)
    equal_parts = inputs.shape[0] // num_group
    inputs = inputs.to(device)
    targets = targets.to(device)
    gradg_list = []

    if grad_mode == 0:
        for i in range(num_group):
            _outputs = net.forward(inputs[i * equal_parts:(i + 1) * equal_parts]) / T
            _loss = F.cross_entropy(_outputs, targets[i * equal_parts:(i + 1) * equal_parts])
            _grad = autograd.grad(_loss, weights, create_graph=True)
            _gz = 0
            _layer = 0
            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    _gz += _grad[_layer].pow(2).sum()  # g * g
                    _layer += 1
            gradg_list.append(autograd.grad(_gz, weights))
    elif grad_mode == 1:
        _grad_ls = []
        for i in range(num_group):
            _outputs = net.forward(inputs[i * equal_parts:(i + 1) * equal_parts]) / T
            _loss = F.cross_entropy(_outputs, targets[i * equal_parts:(i + 1) * equal_parts])
            _grad_ls.append(autograd.grad(_loss, weights, create_graph=True))
        _grad_and = []
        _layer = 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                _gand = 0
                for i in range(num_group):
                    _gand += _grad_ls[i][_layer]
                _grad_and.append(_gand)
                _layer += 1
        for i in range(num_group):
            _gz = 0
            _layer = 0
            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    _gz += (_grad_and[_layer] * _grad_ls[i][_layer]).sum()  # ga * gn
                    _layer += 1
            gradg_list.append(autograd.grad(_gz, weights, retain_graph=True))
    elif grad_mode == 2:
        _grad_ls = []
        for i in range(num_group):
            _outputs = net.forward(inputs[i * equal_parts:(i + 1) * equal_parts]) / T
            _loss = F.cross_entropy(_outputs, targets[i * equal_parts:(i + 1) * equal_parts])
            _grad_ls.append(autograd.grad(_loss, weights, create_graph=True))

        for i in range(num_group):
            for j in range(i + 1, num_group):
                _gz = 0
                _layer = 0
                for layer in net.modules():
                    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                        _gz += (_grad_ls[i][_layer] * _grad_ls[j][_layer]).sum()  # g1 * g2
                        _layer += 1
                gradg_list.append(autograd.grad(_gz, weights, retain_graph=True))
    elif grad_mode == 3:
        for i in range(num_group):
            _outputs = net.forward(inputs[i * equal_parts:(i + 1) * equal_parts]) / T
            _loss = F.cross_entropy(_outputs, targets[i * equal_parts:(i + 1) * equal_parts])
            gradg_list.append(autograd.grad(_loss, weights, retain_graph=True))

    return gradg_list


def reset_mask(net):
    keep_masks = dict()
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            keep_masks[layer] = torch.ones_like(layer.weight.data).float()
    return keep_masks


def get_connected_scores(keep_masks, info='', mode=0, verbose=0):
    _connected_scores = 0
    _last_filter = None
    for m, g in keep_masks.items():
        if isinstance(m, nn.Conv2d) and 'padding' in str(m):  # 不考虑resnet的shortcut部分
            # [n, c, k, k]
            _2d = np.sum(np.abs(keep_masks[m].cpu().detach().numpy()), axis=(2, 3))
            _channel = np.sum(_2d, axis=0)  # 向量

            if _last_filter is not None:  # 第一层不考虑
                for i in range(_channel.shape[0]):  # 遍历通道过滤器
                    if _last_filter[i] == 0 and _channel[i] != 0:
                        _connected_scores += 1
                    if mode == 1:
                        if _last_filter[i] != 0 and _channel[i] == 0:
                            _connected_scores += 1

            _last_filter = np.sum(_2d, axis=1)

    if verbose == 1:
        print(f'{info}-{mode}->connected scores: {_connected_scores}')
    return _connected_scores


def ranking_mask(scores, keep_ratio, normalize=True, eff_rank=False, verbose=0, acc_score=None, oir_mask=None):
    eps = 1e-10
    keep_masks = dict()
    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in scores.values()])
    if oir_mask is not None:
        min_score = torch.min(all_scores)
        # 将已经删除的分数调至最小，避免再次选择
        for m, g in scores.items():
            scores[m][oir_mask[m] == 0] = min_score
            all_scores = torch.cat([torch.flatten(x) for x in scores.values()])
    norm_factor = torch.abs(torch.sum(all_scores)) + eps
    norm_factor = norm_factor if normalize else torch.ones_like(norm_factor)
    all_scores.div_(norm_factor)
    num_params_to_rm = int(len(all_scores) * keep_ratio)
    threshold, _index = torch.topk(all_scores, num_params_to_rm)
    acceptable_score = threshold[-1] if acc_score is None else acc_score
    for m, g in scores.items():
        if eff_rank:
            keep_masks[m] = ((g / norm_factor) != 0).float()
        else:
            keep_masks[m] = ((g / norm_factor) >= acceptable_score).float()
    if verbose == 1:
        print("** norm factor:", norm_factor)
        print('** accept: ', acceptable_score)
    elif verbose == 2:
        return keep_masks, acceptable_score
    elif verbose == 3:
        return keep_masks, acceptable_score, scores
    return keep_masks


def Single_ranking_pruning(net, ratio, samples, device, config=None, reinit=False, retrun_inf=0, verbose=1, oir_mask=None, oir_w=None):
    """
    :param retrun_inf: 其他返回参数，0-返回连通度，1-返回排序分值，2-分值和阈值, 9-只返回分值
    """
    if ratio == 0:
        return reset_mask(net), 0

    keep_ratio = (1 - ratio)
    old_net = net
    net = copy.deepcopy(net)  # .eval()
    net.train() if config.train_mode == 1 else net.eval()
    net.zero_grad()

    num_iters = config.get('num_iters', 1)
    score_mode = config.score_mode

    if score_mode == 0:
        return reset_mask(net), 0

    gradg_list = None
    for it in range(num_iters):
        if verbose == 1:
            print("Iterations %d/%d." % (it, num_iters))
        sample_n = (samples[0][it], samples[1][it]) if isinstance(samples, tuple) else samples
        _hessian_grad = hessian_gradient_product(net, sample_n, device, config, reinit=reinit)
        if gradg_list is None:
            gradg_list = _hessian_grad
        else:
            for i in range(len(gradg_list)):
                _grad_i = _hessian_grad[i]
                gradg_list[i] = [gradg_list[i][_l] + _grad_i[_l] for _l in range(len(_grad_i))]
    # print(len(gradg_list))

    # === 剪枝部分 ===
    """
        score_mode:
            1 - 和
            2 - 和绝对值
            3 - 乘积
            4 - 欧氏距离

        gss: --data_mode 1 --grad_mode 2 --score_mode 2 --num_group 2
        gss-group: --data_mode 1 --grad_mode 2 --score_mode 4 --num_group 5
        grasp: --data_mode 0 --grad_mode 0 --score_mode 1 --num_group 1
        snip: --data_mode 0 --grad_mode 3 --score_mode 2 --num_group 1
    """

    # === 计算分值 ===
    layer_cnt = 0
    grads = dict()
    old_modules = list(old_net.modules())
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            kxt = 0
            weight = oir_w[old_modules[idx]] if oir_w is not None else layer.weight.data
            if score_mode == 1:
                for i in range(len(gradg_list)):
                    _qhg = weight * gradg_list[i][layer_cnt]  # theta_q grad
                    kxt += _qhg
            elif score_mode == 2:
                for i in range(len(gradg_list)):
                    _qhg = weight * gradg_list[i][layer_cnt]  # theta_q grad
                    kxt += _qhg
                kxt = torch.abs(kxt)
            elif score_mode == 3:
                kxt = 1e6  # 约等于超参，估计值，kxt是👴
                for i in range(len(gradg_list)):
                    _qhg = weight * gradg_list[i][layer_cnt]  # theta_q grad
                    kxt *= torch.abs(_qhg)  # 最后线性层有bug，？，不解
            elif score_mode == 4:
                aef = 1e6  # 约等于超参，估计值
                for i in range(len(gradg_list)):
                    _qhg = weight * gradg_list[i][layer_cnt] * aef  # theta_q grad
                    kxt += _qhg.pow(2)
                kxt = kxt.sqrt()
            if isinstance(layer, nn.Conv2d):
                if 'filter' in config.debug:
                    # (n, c, k, k)
                    _s = kxt.shape
                    kxt = torch.mean(kxt, dim=(1, 2, 3), keepdim=True).repeat(1, _s[1], _s[2], _s[3])
                elif 'channel' in config.debug:
                    _s = kxt.shape
                    kxt = torch.mean(kxt, dim=(0, 2, 3), keepdim=True).repeat(_s[0], 1, _s[2], _s[3])
            # 评估分数
            grads[old_modules[idx]] = kxt
            layer_cnt += 1

    if retrun_inf == 9:
        return grads

    # === get masks ===
    keep_masks, threshold = ranking_mask(grads, keep_ratio, verbose=2, oir_mask=oir_mask)
    if verbose == 1:
        print('** accept: ', threshold)
        print('Remaining:', torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))

    _connected_scores = get_connected_scores(keep_masks, f"{'-' * 20}\nBefore", 1, verbose)

    if retrun_inf == 0:
        return keep_masks, _connected_scores
    elif retrun_inf == 2:
        return keep_masks, grads, threshold
    else:
        return keep_masks, grads


def Single_ranking_pruning2(net, ratio, samples, device, config=None, reinit=False, retrun_inf=0, verbose=1, oir_mask=None, oir_w=None):
    """
    :param retrun_inf: 其他返回参数，0-返回连通度，1-返回排序分值，2-分值和阈值, 9-只返回分值
    """
    scores = None
    keep_ratio = (1 - ratio)
    # 采样
    inputs = []
    targets = []
    for it in range(config.get('num_iters', 1)):
        x, y = fetch_data(samples, config.classe, config.samples_per_class)
        inputs.append(x)
        targets.append(y)

    if 'group' in config.debug:
        # scores = Single_ranking_pruning(net, ratio, (inputs, targets), device, config, False, retrun_inf=9)

        for i in range(10):

            # # 不同标签组
            # _k=random.randint(0, 9)*10
            # _x, _y = [], []
            # _x.append(inputs[0][_k+5:_k+15])
            # _y.append(targets[0][_k+5:_k+15])
            # samples = (_x, _y)
            # _s = Single_ranking_pruning(net, ratio, samples, device, config, False, retrun_inf=9)
            # if scores is None:
            #     scores = _s
            # else:
            #     for m, s in _s.items():
            #         scores[m] = scores[m] + s

            # 同标签组
            # samples[0][0], samples[1][0] = inputs[0][i*10:i*10+10], targets[0][i*10:i*10+10]
            _x, _y = [], []
            _x.append(inputs[0][i*10:i*10+10])
            _y.append(targets[0][i*10:i*10+10])
            samples = (_x, _y)
            _s = Single_ranking_pruning(net, ratio, samples, device, config, False, retrun_inf=9)
            # if scores is None:
            #     scores = _s
            # else:
            #     for m, s in _s.items():
            #         if 'and' in config.debug:
            #             scores[m] = scores[m] + s
            #         elif 'sub' in config.debug:
            #             scores[m] = scores[m] - s
            #         elif 'mul' in config.debug:
            #             scores[m] = scores[m] * s
            #         elif 'div' in config.debug:
            #             scores[m] = scores[m] / s
            #         if 'abs' in config.debug:
            #             scores[m] = torch.abs(scores[m])

            # for m, s in _s.items():
            #     scores[m] = scores[m] - s / 10
            if scores is None:
                scores = _s
            else:
                for m, s in _s.items():
                    scores[m] = scores[m] + s

    # === 这里做出修改 采用均值处理 ===
    elif 'mean' in config.debug:
        scores = dict()
        s_max = None
        s_min = None

        # 不同标签组
        for i in range(10):
            for j in range(i + 1, 10):
                _index = np.zeros(100)
                _index[i*10:i*10+5] = 1
                _index[j*10:j*10+5] = 1
                _index = np.array(_index, dtype= bool)
                _x, _y = [], []
                _x.append(inputs[0][_index])
                _y.append(targets[0][_index])
                # print(targets[0][_index])
                samples = (_x, _y)
                _s = Single_ranking_pruning(net, ratio, samples, device, config, False, retrun_inf=9)
                if s_max is None:
                    s_max = _s
                else:
                    for m, s in _s.items():
                        s_max[m] = s_max[m] + s

        # 同标签组
        for i in range(10):
            # 同标签组
            samples[0][0], samples[1][0] = inputs[0][i*10:i*10+10], targets[0][i*10:i*10+10]
            _s = Single_ranking_pruning(net, ratio, samples, device, config, False, retrun_inf=9)
            if s_min is None:
                s_min = _s
            else:
                for m, s in _s.items():
                    s_min[m] = s_min[m] + s

        # 计算分数
        for m, s in s_min.items():
            if 'and' in config.debug:
                scores[m] = s_max[m]/45 + s_min[m]/10
            elif 'sub' in config.debug:
                scores[m] = s_max[m]/45 - s_min[m]/10
            elif 'mul' in config.debug:
                scores[m] = s_max[m]*s_min[m]
            elif 'div' in config.debug:
                scores[m] = s_max[m]/s_min[m]
    else:
        scores = dict()
        # === label 减 data ===
        samples = (inputs, targets)
        _score0 = Single_ranking_pruning(net, ratio, samples, device, config, False, retrun_inf=9)

        _index = []
        samples_per_class, num_classes = 10, config.classe
        for i in range(samples_per_class):
            _index.extend([i + j * samples_per_class for j in range(0, num_classes)])
        inputs[0] = inputs[0][_index]
        targets[0] = targets[0][_index]
        _score1 = Single_ranking_pruning(net, ratio, samples, device, config, False, retrun_inf=9)

        for m, s in _score0.items():
            if 'and' in config.debug:
                scores[m] = _score0[m] + _score1[m]
            elif 'sub' in config.debug:
                scores[m] = _score0[m] - _score1[m]
            elif 'mul' in config.debug:
                scores[m] = _score0[m] * _score1[m]
            elif 'div' in config.debug:
                scores[m] = _score0[m] / _score1[m]
            if 'abs' in config.debug:
                scores[m] = torch.abs(scores[m])


    # === get masks ===
    # for m, s in scores.items():
    #     scores[m] = scores[m].abs_()
    keep_masks, threshold = ranking_mask(scores, keep_ratio, verbose=2, oir_mask=oir_mask)
    if verbose == 1:
        print('** accept: ', threshold)
        print('Remaining:', torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))

    _connected_scores = get_connected_scores(keep_masks, f"{'-' * 20}\nBefore", 1, verbose)

    if retrun_inf == 0:
        return keep_masks, _connected_scores
    elif retrun_inf == 2:
        return keep_masks, scores, threshold
    else:
        return keep_masks, scores


# Based on https://github.com/ganguli-lab/Synaptic-Flow/blob/master/Pruners/pruners.py#L178
def SynFlow(model, ratio, dataloader, device, num_iters, eff_rank=False, ori_masks=None, dynamic=False):

    @torch.no_grad()
    def linearize(model):
        signs = {}
        for name, param in model.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(model, signs):
        for name, param in model.state_dict().items():
            param.mul_(signs[name])

    old_net = model
    net = copy.deepcopy(model)
    net.eval()
    net.zero_grad()
    modules_ls = list(old_net.modules())
    signs = linearize(net)
    (data, _) = dataloader if isinstance(dataloader, tuple) else next(iter(dataloader))
    input_dim = list(data[0, :].shape)
    input = torch.ones([1] + input_dim).to(device)
    copy_net_weights = dict()
    for layer in old_net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            copy_net_weights[layer] = torch.clone(layer.weight.data)
    keep_masks = reset_mask(old_net) if ori_masks is None else ori_masks
    grads = dict()

    desc = ('[keep ratio=%s] acceptable score: %e' % (1, 0))
    prog_bar = tqdm(range(num_iters), total=num_iters, desc=desc, leave=True) if num_iters > 1 else range(num_iters)
    for epoch in prog_bar:
        keep_ratio = (1.0 - ratio)**((epoch + 1) / num_iters)
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight.data = (copy_net_weights[modules_ls[idx]] * keep_masks[modules_ls[idx]]).abs_()
        net.zero_grad()
        # forward
        output = net(input)
        torch.sum(output).backward()
        # synflow score
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                _w = copy_net_weights[modules_ls[idx]] if dynamic else layer.weight.data
                grads[modules_ls[idx]] = (_w * layer.weight.grad).abs_()  # theta g
        # synflow masks
        keep_masks, acceptable_score = ranking_mask(grads, keep_ratio, False, eff_rank, 2)

        if num_iters > 1:
            desc = ('[keep ratio=%s] acceptable score: %e' % (keep_ratio, acceptable_score))
            prog_bar.set_description(desc, refresh=True)

    # nonlinearize(net, signs)

    return keep_masks, grads


# Based on https://github.com/avysogorets/Effective-Sparsity/blob/main/effective_masks.py#28
def effective_masks_synflow(model, masks, trainloader, device):
    """ computes effective sparsity of a pruned model using SynFlow
    """
    true_masks, _ = SynFlow(model, 0, trainloader, device, 1, True, masks)
    return true_masks


def coincide_mask(mask1, mask2):
    _coin_num = 0
    for m, s in mask1.items():
        _coin_num += torch.sum(mask1[m]*mask2[m] == 1)
    _m2_num = torch.sum(torch.cat([torch.flatten(x == 1) for x in mask2.values()]))
    return (_coin_num/_m2_num).cpu().detach().numpy()


def get_keep_ratio(masks, layer_ratio=False, verbose=0):
    _l_r = []
    _remain_num = 0
    _all_num = 0
    for m, s in masks.items():
        _re_num = torch.sum(masks[m] == 1)
        _a_num = masks[m].numel()
        _remain_num += _re_num
        _all_num += _a_num
        if layer_ratio:
            _l_r.append((_re_num / _a_num).cpu().detach().numpy())
    if verbose == 1:
        print(_l_r)
    if layer_ratio:
        return (_remain_num / _all_num).cpu().detach().numpy(), _l_r
    else:
        return (_remain_num / _all_num).cpu().detach().numpy()


def Iterative_pruning(model, ratio, trainloader, device, config):

    old_net = model
    old_modules = list(old_net.modules())
    keep_masks = reset_mask(old_net)
    net = copy.deepcopy(model)
    net.train()
    copy_net_weights = dict()
    for layer in old_net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            copy_net_weights[layer] = torch.clone(layer.weight.data)

    num_iters = config.num_iters_prune
    desc = ('[keep ratio=%s] acceptable score: %e' % (1, 0))
    prog_bar = tqdm(range(num_iters), total=num_iters, desc=desc, leave=True) if num_iters > 1 else range(num_iters)
    for epoch in prog_bar:
        # keep_ratio = 1.0 - ratio * ((epoch + 1) / num_iters)
        keep_ratio = (1.0 - ratio) ** ((epoch + 1) / num_iters)

        net.zero_grad()
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight.data = copy_net_weights[old_modules[idx]] * keep_masks[old_modules[idx]]

        score_mode = config.score_mode
        gradg_list = hessian_gradient_product(net, trainloader, device, config)

        layer_cnt = 0
        grads = dict()
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                kxt = 0
                _w = copy_net_weights[old_modules[idx]] if config.dynamic else layer.weight.data
                if score_mode == 1:
                    for i in range(len(gradg_list)):
                        _qhg = _w * gradg_list[i][layer_cnt]  # theta_q grad
                        kxt += _qhg
                elif score_mode == 2:
                    for i in range(len(gradg_list)):
                        _qhg = _w * gradg_list[i][layer_cnt]  # theta_q grad
                        kxt += _qhg
                    kxt = torch.abs(kxt)
                elif score_mode == 3:
                    kxt = 1e6
                    for i in range(len(gradg_list)):
                        _qhg = _w * gradg_list[i][layer_cnt]  # theta_q grad
                        kxt *= torch.abs(_qhg)
                elif score_mode == 4:
                    aef = 1e6
                    for i in range(len(gradg_list)):
                        _qhg = _w * gradg_list[i][layer_cnt] * aef  # theta_q grad
                        kxt += _qhg.pow(2)
                    kxt = kxt.sqrt()
                if isinstance(layer, nn.Conv2d):
                    if 'filter' in config.debug:
                        # (n, c, k, k)
                        _s = kxt.shape
                        kxt = torch.mean(kxt, dim=(1, 2, 3), keepdim=True).repeat(1, _s[1], _s[2], _s[3])
                    elif 'channel' in config.debug:
                        _s = kxt.shape
                        kxt = torch.mean(kxt, dim=(0, 2, 3), keepdim=True).repeat(_s[0], 1, _s[2], _s[3])
                grads[old_modules[idx]] = kxt
                layer_cnt += 1

        oir_mask = None if config.dynamic else keep_masks
        keep_masks, threshold = ranking_mask(grads, keep_ratio, verbose=2, oir_mask=oir_mask)
        if 'a' in config.debug:
            _score = kernel_link_score(grads, keep_masks)
            keep_masks, threshold = ranking_mask(_score, keep_ratio, verbose=2)

        if num_iters > 1:
            desc = ('[keep ratio=%s] acceptable score: %e' % (keep_ratio, threshold))
            prog_bar.set_description(desc, refresh=True)

    return keep_masks, 0


# 迭代分析
def Iterative_pruning_figure(model, ratio, trainloader, device, config, num_iters):

    old_net = model
    old_modules = list(old_net.modules())
    keep_masks = reset_mask(old_net)
    last_masks = reset_mask(old_net)
    mask_key = [x for x in keep_masks.keys()]
    net = copy.deepcopy(model)
    net.train()
    copy_net_weights = dict()
    for layer in old_net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            copy_net_weights[layer] = torch.clone(layer.weight.data)

    figure = PltScore(600, 1000, 100)
    keep_gtg_ls = []
    remove_gtg_ls = []
    keep_remove_gtg_ls = []
    newr_gtg_ls = []
    ghg_ls = []
    keep_ratio_ls = []
    layer_ratio_ls = []
    coin_ls = []
    exp_ratio_ls = []
    last_grad = dict()
    grad_rise_prop = [0]
    remove_prop = [0]
    newly_remove_prop = [0]
    keep_remove_num_prop = [0]

    std_ls = []
    mean_ls = []

    invert_flag = 0
    last_gtg = 0
    last_grad = None

    test_gtg_ls = []  # 选取新移除权重，其余置零

    data_mode = config.data_mode
    grad_mode = config.grad_mode
    num_group = config.num_group
    score_mode = config.score_mode
    samples_per_class = config.samples_per_class
    num_classes = config.classe

    inputs, targets = fetch_data(trainloader, num_classes, samples_per_class, dm=data_mode)
    inputs = inputs.to(device)
    targets = targets.to(device)

    if 'p' in config.debug:
        import matplotlib.pyplot as plt
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.subplots()
        color = ['red', 'orange', 'yellow', 'green', 'blue', 'cyan', 'purple', 'pink', 'coral', 'gold', 'lime', 'navy',
                 'teal', 'indigo']
        index = None


    desc = ('[keep ratio=%s] acceptable score: %e' % (1, 0))
    prog_bar = tqdm(range(num_iters), total=num_iters, desc=desc, leave=True) if num_iters > 1 else range(num_iters)
    for epoch in prog_bar:
        keep_ratio = (1.0 - ratio) ** ((epoch + 1) / num_iters)

        net.zero_grad()
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight.data = copy_net_weights[old_modules[idx]] * keep_masks[old_modules[idx]]
                # layer.weight.data = copy_net_weights[old_modules[idx]]  # 恢复

        weights = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                weights.append(layer.weight)
        for w in weights:
            w.requires_grad_(True)

        # inputs, targets = fetch_data(trainloader, num_classes, samples_per_class, dm=data_mode)
        # inputs = inputs.to(device)
        # targets = targets.to(device)
        gradg_list = []
        gtg_ls = []
        keep_gtg, remove_gtg = [], []
        newr_gtg = []
        keep_remove_gtg = []
        _grad_rise = 0
        _remove_grad_rise = 0
        _newly_remove_gr = 0
        _remove_num = 0
        _newly_remove_num = 0
        grad_score = dict()

        _outputs = net.forward(inputs) / 200
        _loss = F.cross_entropy(_outputs, targets)
        _grad = autograd.grad(_loss, weights, create_graph=True)
        # _loss.backward(create_graph=True, retain_graph=True)
        _gz = 0
        _gk, _gr = 0, 0
        _layer = 0
        _ga = 0
        _gs = 0
        _std = 0
        _mean = 0
        temp_masks = dict()
        temp_sum = 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                # _g = layer.weight.grad
                _g = _grad[_layer]
                # grad_score[mask_key[_layer]] = torch.abs(_g)
                grad_score[mask_key[_layer]] = torch.abs(_g)*(1-keep_masks[mask_key[_layer]])  # 置零部分
                # grad_score[mask_key[_layer]] = _g

                _gz += _g.pow(2).sum()  # g * g

                # == 忽略零值通道gtg ==
                # if isinstance(layer, nn.Conv2d):
                #     _s = _g.shape
                #     _ind = torch.mean(keep_masks[mask_key[_layer]], dim=(0, 2, 3), keepdim=True).repeat(_s[0], 1, _s[2], _s[3]) > 0
                #     _gz += _g[_ind].pow(2).sum()
                # else:
                #     _gz += _g.pow(2).sum()

                # if epoch > 1:
                #     _ind = (torch.abs((_grad[_layer]) - torch.abs(last_grad[_layer])) != 0)
                #     # print(torch.sum(_ind == False))
                #     temp_sum += torch.sum(_ind[_grad[_layer]!=0] == False)
                # if invert_flag == 1:
                #     # _ind = (torch.abs((_grad[_layer]) - torch.abs(last_grad[_layer])) != 0)
                #     # print(torch.sum(_ind == False))
                #     # temp_sum += torch.sum(_ind == False)
                #     _gz += _g[_ind].pow(2).sum()
                # else:
                #     _gz += _g.pow(2).sum()  # g * g

                _gk += (_g*keep_masks[mask_key[_layer]]).pow(2).sum()
                _gr += (_g*(1-keep_masks[mask_key[_layer]])).pow(2).sum()

                # _change_ind = (keep_masks[mask_key[_layer]].bool())^(last_masks[mask_key[_layer]].bool())  # 有变动的权重
                # _new_remove_ind = (keep_masks[mask_key[_layer]]-last_masks[mask_key[_layer]]) < 0  # 移除的权重
                _new_remove_ind = (1-keep_masks[mask_key[_layer]]).bool()  # 移除的权重
                _keep_remove_ind = ((~(keep_masks[mask_key[_layer]].bool()))&(~(last_masks[mask_key[_layer]].bool())))  # 删减部分未变动的权重

                # _gs += (_grad[_layer]*((keep_masks[mask_key[_layer]].bool())^(last_masks[mask_key[_layer]].bool())).float()).pow(2).sum()  # 有变动的权重
                _ga += (_g*_new_remove_ind.float()).pow(2).sum()  # 移除的权重
                _gs += (_g*_keep_remove_ind.float()).pow(2).sum()  # 删减部分未变动的权重

                # _s = _grad[(1-keep_masks[mask_key[_layer]]).bool()].std()
                # _m = _grad[(1-keep_masks[mask_key[_layer]]).bool()].mean()
                _s = (_g[(1-keep_masks[mask_key[_layer]]).bool()]).pow(2).std()
                _m = (_g[(1-keep_masks[mask_key[_layer]]).bool()]).pow(2).mean()
                _std += _s
                _mean += _m

                # temp_masks[m] = (((keep_masks[m]) - (last_masks[m])) < 0).float()

                # # 检查梯度变化
                # if epoch > 0:
                #     _detla = (torch.abs(_grad[_layer]) - torch.abs(last_grad[mask_key[_layer]])) > 0
                #     _grad_rise += torch.sum(_detla)
                #     _remove_grad_rise += torch.sum(_detla[_keep_remove_ind])
                #     _remove_num += torch.sum(_keep_remove_ind)
                #     _newly_remove_gr += torch.sum(_detla[_new_remove_ind])
                #     _newly_remove_num += torch.sum(_new_remove_ind)
                # last_grad[mask_key[_layer]] = _grad[_layer]

                _layer += 1
        # print(temp_sum)
        gradg_list.append(autograd.grad(_gz, weights))
        gtg_ls.append(_gz)
        keep_gtg.append(_gk)
        remove_gtg.append(_gr)
        newr_gtg.append(_ga)
        keep_remove_gtg.append(_gs)

        last_grad = _grad
        # if invert_flag == 0 and last_gtg > _gz:
        # if invert_flag == 0 and keep_ratio < 0.01:
        #     invert_flag = 1
        # last_gtg = _gz

        # 二阶项 gHg
        # layer_cnt = 0
        # gHg = 0
        # for layer in net.modules():
        #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #         _grad = layer.weight.grad
        #         gHg += (gradg_list[0][layer_cnt] * _grad).sum()
        #         layer_cnt += 1
        # ghg_ls.append(gHg.cpu().detach().numpy())
        # del gHg

        layer_cnt = 0
        grads = dict()
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                kxt = 0
                if invert_flag == 1:
                    _wh = layer.weight.data
                else:
                    _wh = copy_net_weights[old_modules[idx]] if config.dynamic else layer.weight.data
                if score_mode == 1:
                    for i in range(len(gradg_list)):
                        _qhg = _wh * gradg_list[i][layer_cnt]  # theta_q grad
                        kxt += _qhg
                elif score_mode == 2:
                    for i in range(len(gradg_list)):
                        _qhg = _wh * gradg_list[i][layer_cnt]  # theta_q grad
                        kxt += _qhg
                    kxt = torch.abs(kxt)
                elif score_mode == 3:
                    kxt = 1e6
                    for i in range(len(gradg_list)):
                        _qhg = _wh * gradg_list[i][layer_cnt]  # theta_q grad
                        kxt *= torch.abs(_qhg)
                elif score_mode == 4:
                    aef = 1e6
                    for i in range(len(gradg_list)):
                        _qhg = _wh * gradg_list[i][layer_cnt] * aef  # theta_q grad
                        kxt += _qhg.pow(2)
                    kxt = kxt.sqrt()
                if invert_flag == 1:
                    kxt = -kxt
                grads[old_modules[idx]] = kxt
                layer_cnt += 1

        last_masks = keep_masks
        if invert_flag == 1:
            # grads = kernel_link_score(grads, keep_masks)
            keep_masks, threshold = ranking_mask(grads, keep_ratio, verbose=2, oir_mask=keep_masks)
        else:
            keep_masks, threshold = ranking_mask(grads, keep_ratio, verbose=2)

        # keep_masks, threshold = Single_ranking_pruning(model, 1-keep_ratio, ([inputs], [targets]), device, config, reinit=False, retrun_inf=0, verbose=2)

        # if 'p' in config.debug:
        #     for l, (m, s) in enumerate(grads.items()):
        #         if l == 1:
        #             if epoch == 0:
        #                 _s, index = torch.sort(torch.cat([torch.flatten(s)]), descending=True)
        #                 np_s = _s.cpu().detach().numpy()
        #                 plt.plot(range(1, len(np_s) + 1, 1), np_s, color=color[epoch%10], label=f'{(1 - keep_ratio) * 100:0.2f}')
        #                 plt.axvline(np.argwhere(np_s > min(np_s))[-1], color=color[epoch%10], linestyle=':')
        #             else:
        #                 if keep_ratio < 0.06:
        #                     np_s = torch.flatten(s)[index].cpu().detach().numpy()
        #                     plt.scatter(range(1, len(np_s) + 1, 1), np_s, color=color[epoch%10], s=1, label=f'{(1 - keep_ratio) * 100:0.2f}')

        true_masks = effective_masks_synflow(model, keep_masks, (inputs, targets), device)
        _coin = 1-get_keep_ratio(true_masks)
        # _coin = coincide_mask(last_masks, keep_masks)  # 重合度
        _r, _l_r_ls = get_keep_ratio(keep_masks, True)

        keep_gtg_ls.append(keep_gtg[0].cpu().detach().numpy())
        remove_gtg_ls.append(remove_gtg[0].cpu().detach().numpy())
        keep_ratio_ls.append(1-keep_ratio)
        layer_ratio_ls.append(_l_r_ls)
        newr_gtg_ls.append(newr_gtg[0].cpu().detach().numpy())
        keep_remove_gtg_ls.append(keep_remove_gtg[0].cpu().detach().numpy())
        std_ls.append(_std.cpu().detach().numpy())
        mean_ls.append(_mean.cpu().detach().numpy())
        coin_ls.append(_coin)
        # if epoch > 0:
        #     all_num = sum([x.numel() for x in keep_masks.values()])
        #     grad_rise_prop.append((_grad_rise/all_num).cpu().detach().numpy())
        #     remove_prop.append((_remove_grad_rise/_remove_num).cpu().detach().numpy())
        #     newly_remove_prop.append((_newly_remove_gr/_newly_remove_num).cpu().detach().numpy())
        #     keep_remove_num_prop.append((_remove_num/all_num).cpu().detach().numpy())

        # 梯度分析
        # grad_mask, threshold = ranking_mask(grad_score, 0.02, verbose=2)
        # _r, _l_r_ls = get_keep_ratio(grad_mask, True)
        # layer_ratio_ls.append(_l_r_ls)
        # del grad_mask, grad_score

        if num_iters > 1:
            desc = ('[keep ratio=%s] acceptable score: %e' % (keep_ratio, threshold))
            prog_bar.set_description(desc, refresh=True)

        # if keep_ratio > 0.02 and keep_ratio < 0.04 :
        #     keep_masks[mask_key[9]][0,:,:,:] *= 0
        # if keep_ratio > 0.1:
        #     test_gtg_ls.append(0)
        # else:
        # inputs, targets = fetch_data(trainloader, num_classes, samples_per_class, dm=data_mode)
        # inputs = inputs.to(device)
        # targets = targets.to(device)
        # temp_masks = dict()
        # for m, s in keep_masks.items():
        #     temp_masks[m] = (((keep_masks[m])-(last_masks[m]))<0).float()
        # exp_ratio_ls.append(get_keep_ratio(temp_masks))
        # net.zero_grad()
        # for idx, layer in enumerate(net.modules()):
        #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #         layer.weight.data = copy_net_weights[old_modules[idx]] * temp_masks[old_modules[idx]]
        #
        # _outputs = net.forward(inputs) / 200
        # _loss = F.cross_entropy(_outputs, targets)
        # _loss.backward()
        # _gx = 0
        # _layer = 0
        # for layer in net.modules():
        #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #         # _gx += (layer.weight.grad * (1-temp_masks[mask_key[_layer]])).pow(2).sum()
        #         _gx += (layer.weight.grad).pow(2).sum()
        #         _layer += 1
        # test_gtg_ls.append(_gx.cpu().detach().numpy())

    if 'p' in config.debug:
        plt.xscale('log')
        plt.axhline(0, color='gray', linestyle=':')
        plt.legend(loc='upper right')
        plt.show()

    fig_num = 3

    # gtg
    ax1 = figure.plt.subplot(fig_num, 1, 1)
    ax1.plot(keep_ratio_ls, keep_gtg_ls, color=figure.color[0], linestyle='--', marker='o', label='keep')
    ax1.plot(keep_ratio_ls, remove_gtg_ls, color=figure.color[1], linestyle='--', marker='o', label='remove')
    ax1.plot(keep_ratio_ls, newr_gtg_ls, color=figure.color[2], linestyle='--', marker='o', label='newly removed')
    ax1.plot(keep_ratio_ls, keep_remove_gtg_ls, color=figure.color[3], linestyle='--', marker='o', label='continuous removed')
    # ax1.plot(keep_ratio_ls, test_gtg_ls, color=figure.color[4], linestyle='--', marker='o', label='exp')
    ax1.axhline(0, color='gray', linestyle=':')
    ax1.legend()

    # grad
    # ax2 = figure.plt.subplot(fig_num, 1, 2)
    # ax2.plot(keep_ratio_ls, grad_rise_prop, color=figure.color[4], linestyle='--', marker='o', label='the proportion of gradient increase')
    # ax2.plot(keep_ratio_ls, newly_remove_prop, color=figure.color[5], linestyle='--', marker='o', label='newly removed')
    # ax2.plot(keep_ratio_ls, remove_prop, color=figure.color[6], linestyle='--', marker='o', label='continuous removed')
    # ax2.plot(keep_ratio_ls, keep_remove_num_prop, color=figure.color[7], linestyle='--', marker='o', label='removed quantity ratio')
    # ax2.legend()

    # std mean
    # ax2 = figure.plt.subplot(fig_num, 1, 2)
    # ax2.plot(keep_ratio_ls, std_ls, color=figure.color[0], linestyle='--', marker='o', label='std')
    # ax2.plot(keep_ratio_ls, mean_ls, color=figure.color[1], linestyle='--', marker='o', label='mean')
    # ax2.axhline(0, color='gray', linestyle=':')
    # ax2.legend()

    # ghg
    # ax3 = figure.plt.subplot(fig_num, 1, 3)
    # ax3.plot(keep_ratio_ls, ghg_ls, color=figure.color[0], linestyle='--', marker='o', label='ghg')
    # ax3.legend()

    # 有效压缩
    ax3 = figure.plt.subplot(fig_num, 1, 2)
    ax3.plot(keep_ratio_ls, coin_ls, color=figure.color[0], linestyle='--', marker='o', label='real ratio')
    ax3.plot(keep_ratio_ls, keep_ratio_ls, color=figure.color[1], linestyle='--', marker='o', label='target ratio')
    # ax3.plot(keep_ratio_ls, exp_ratio_ls, color=figure.color[2], linestyle='--', marker='o', label='exp ratio')
    ax3.axhline(1, color='gray', linestyle=':')
    ax3.legend()

    # 有效压缩比
    ax3 = figure.plt.subplot(fig_num, 1, 3)
    _prop = [(coin_ls[i]-keep_ratio_ls[i])/(1-keep_ratio_ls[i]) for i in range(len(coin_ls))]
    ax3.plot(keep_ratio_ls, _prop, color=figure.color[0], linestyle='--', marker='o', label='')
    ax3.axhline(0, color='gray', linestyle=':')
    ax3.legend()

    # ax2 = figure.plt.subplot(fig_num, 1, 3)
    # for i in range(len(layer_ratio_ls)):
    #     if keep_ratio_ls[i] > 0.9:
    #         ax2.plot(range(len(layer_ratio_ls[0])), layer_ratio_ls[i], color=figure.color[i%10], linestyle='--', marker='o', label=f'{keep_ratio_ls[i]:0.3f}')
    # ax2.legend()

    # ax3 = figure.plt.subplot(fig_num, 1, 3)
    # for i in range(len(layer_ratio_ls)):
    #     if keep_ratio_ls[i] > 0.99:
    #         ax3.plot(range(len(layer_ratio_ls[0])), layer_ratio_ls[i], color=figure.color[i%10], linestyle='--', marker='o', label=f'{keep_ratio_ls[i]:0.3f}')
    # ax3.legend()

    figure.plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
    figure.plt.show()

    return keep_masks, 0


def calculate_gradient(net, inputs, targets, weights=None):
    if not weights:
        weights = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                weights.append(layer.weight)
        for w in weights:
            w.requires_grad_(True)
    _outputs = net.forward(inputs) / 200
    _loss = F.cross_entropy(_outputs, targets)
    _grad = autograd.grad(_loss, weights, create_graph=True)
    _gtg = 0
    _layer = 0
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            _g = _grad[_layer]
            _gtg += _g.pow(2).sum()  # g * g
            _layer += 1
    return _grad, _gtg


# 移除权重对网络梯度的影响
def Iterative_pruning_figure2(model, ratio, trainloader, device, config, num_iters):

    old_net = model
    old_modules = list(old_net.modules())
    keep_masks = reset_mask(old_net)
    last_masks = reset_mask(old_net)
    mask_key = [x for x in keep_masks.keys()]
    net = copy.deepcopy(model)
    net.train()
    copy_net_weights = dict()
    for layer in old_net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            copy_net_weights[layer] = torch.clone(layer.weight.data)

    data_mode = config.data_mode
    samples_per_class = config.samples_per_class
    num_classes = config.classe
    inputs, targets = fetch_data(trainloader, num_classes, samples_per_class, dm=data_mode)
    inputs = inputs.to(device)
    targets = targets.to(device)

    # 全网络的梯度和Hg
    weights = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weights.append(layer.weight)
    for w in weights:
        w.requires_grad_(True)
    _grad, _gtg = calculate_gradient(net, inputs, targets, weights)
    # _Hg = autograd.grad(_gtg, weights)
    # layer_cnt = 0
    # grads = dict()
    # for idx, layer in enumerate(net.modules()):
    #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #         _wh = layer.weight.data
    #         _qhg = _wh * _Hg[layer_cnt]  # theta_q grad
    #         grads[old_modules[idx]] = torch.abs(_qhg)
    #         layer_cnt += 1
    # keep_masks, threshold = ranking_mask(grads, 0.02, verbose=2)
    # _r, _l_r_ls = get_keep_ratio(keep_masks, True)

    # 修剪98%的gtg
    # for idx, layer in enumerate(net.modules()):
    #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #         layer.weight.data = copy_net_weights[old_modules[idx]] * keep_masks[old_modules[idx]]
    # _grad_1, _gtg = calculate_gradient(net, inputs, targets)

    # 移除某一层
    # keep_masks[mask_key[0]] *= 0
    remove_layer = 15
    # keep_masks[mask_key[remove_layer]][0, :] *= 0  #
    # keep_masks[mask_key[remove_layer]][:, 0] *= 0  #
    # keep_masks[mask_key[remove_layer]][0, :, :, :] *= 0  # 移除某个过滤器
    keep_masks[mask_key[remove_layer]][:, :, :, :] *= 0  # 移除某个通道
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight.data = copy_net_weights[old_modules[idx]] * keep_masks[old_modules[idx]]
    _grad_2, _gtg = calculate_gradient(net, inputs, targets)

    # 作图
    figure = PltScore(1000, 1000, 100)

    # fig_num = len(mask_key)
    # # fig_num = 9
    # _row = math.ceil(fig_num**0.5)
    # for i in range(fig_num):
    #     ax1 = figure.plt.subplot(_row, _row, i+1)
    #
    #     _g, _ind = torch.sort(torch.cat([torch.flatten(_grad[i])]), descending=True)
    #     np_g = _g.cpu().detach().numpy()
    #     ax1.plot(range(1, len(np_g) + 1, 1), np_g, color=figure.color[0], label=f'全网络')
    #
    #     # np_g = torch.flatten(_grad_1[i])[_ind].cpu().detach().numpy()
    #     # ax1.scatter(range(1, len(np_g) + 1, 1), np_g, color=figure.color[1], s=1, label=f'修剪98%，本层剩余{_l_r_ls[i]*100:.2f}%')
    #     np_g = torch.flatten(_grad_2[i])[_ind].cpu().detach().numpy()
    #     ax1.scatter(range(1, len(np_g) + 1, 1), np_g, color=figure.color[3], s=1, label=f'移除某一层')
    #
    #     ax1.legend()

    ax1 = figure.plt.subplot(1, 2, 1)

    # _g, _ind = torch.sort(torch.cat([torch.flatten(_grad[remove_layer][0, :])]), descending=True)
    # np_g = _g.cpu().detach().numpy()
    # ax1.plot(range(1, len(np_g) + 1, 1), np_g, color=figure.color[0], label=f'全网络')
    # np_g = torch.flatten(_grad_2[remove_layer][0, :])[_ind].cpu().detach().numpy()
    # ax1.scatter(range(1, len(np_g) + 1, 1), np_g, color=figure.color[3], s=1, label=f'移除通道')

    # _g, _ind = torch.sort(torch.cat([torch.flatten(_grad[remove_layer][:, 0])]), descending=True)
    # np_g = _g.cpu().detach().numpy()
    # ax1.plot(range(1, len(np_g) + 1, 1), np_g, color=figure.color[0], label=f'全网络')
    # np_g = torch.flatten(_grad_2[remove_layer][:, 0])[_ind].cpu().detach().numpy()
    # ax1.scatter(range(1, len(np_g) + 1, 1), np_g, color=figure.color[3], s=1, label=f'移除过滤器')

    # _g, _ind = torch.sort(torch.cat([torch.flatten(_grad[remove_layer][0, :, :, :])]), descending=True)
    # np_g = _g.cpu().detach().numpy()
    # ax1.plot(range(1, len(np_g) + 1, 1), np_g, color=figure.color[0], label=f'全网络')
    # np_g = torch.flatten(_grad_2[remove_layer][0, :, :, :])[_ind].cpu().detach().numpy()
    # ax1.scatter(range(1, len(np_g) + 1, 1), np_g, color=figure.color[3], s=1, label=f'移除过滤器')

    _g, _ind = torch.sort(torch.cat([torch.flatten(_grad[remove_layer][:, 0, :, :])]), descending=True)
    np_g = _g.cpu().detach().numpy()
    ax1.plot(range(1, len(np_g) + 1, 1), np_g, color=figure.color[0], label=f'通道的梯度')
    np_g = torch.flatten(_grad_2[remove_layer][:, 0, :, :])[_ind].cpu().detach().numpy()
    ax1.scatter(range(1, len(np_g) + 1, 1), np_g, color=figure.color[3], s=1, label=f'通道置零之后')

    ax1.legend()

    # 观察上下层对应的梯度变化
    # ax2 = figure.plt.subplot(1, 2, 2)
    # _g, _ind = torch.sort(torch.cat([torch.flatten(_grad[remove_layer+1][:, 1, :, :])]), descending=True)
    # np_g = _g.cpu().detach().numpy()
    # ax2.plot(range(1, len(np_g) + 1, 1), np_g, color=figure.color[0], label=f'下一层对应的通道')
    # np_g = torch.flatten(_grad_2[remove_layer+1][:, 1, :, :])[_ind].cpu().detach().numpy()
    # ax2.scatter(range(1, len(np_g) + 1, 1), np_g, color=figure.color[3], s=1, label=f'移除过滤器之后')
    # ax2.legend()

    ax2 = figure.plt.subplot(1, 2, 2)
    _g, _ind = torch.sort(torch.cat([torch.flatten(_grad[remove_layer-1][0, :, :, :])]), descending=True)
    np_g = _g.cpu().detach().numpy()
    ax2.plot(range(1, len(np_g) + 1, 1), np_g, color=figure.color[0], label=f'上一层对应的过滤器')
    np_g = torch.flatten(_grad_2[remove_layer-1][0, :, :, :])[_ind].cpu().detach().numpy()
    ax2.scatter(range(1, len(np_g) + 1, 1), np_g, color=figure.color[3], s=1, label=f'通道置零之后')
    ax2.legend()

    figure.plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    figure.plt.show()

    return keep_masks, 0


# 移除权重对敏感度分值的影响
def Iterative_pruning_figure22(model, ratio, trainloader, device, config, num_iters):

    old_net = model
    old_modules = list(old_net.modules())
    keep_masks = reset_mask(old_net)
    mask_key = [x for x in keep_masks.keys()]
    net = copy.deepcopy(model)
    net.train()
    copy_net_weights = dict()
    for layer in old_net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            copy_net_weights[layer] = torch.clone(layer.weight.data)

    data_mode = config.data_mode
    samples_per_class = config.samples_per_class
    num_classes = config.classe
    inputs, targets = fetch_data(trainloader, num_classes, samples_per_class, dm=data_mode)
    inputs = inputs.to(device)
    targets = targets.to(device)

    # 全网络的梯度和Hg
    weights = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weights.append(layer.weight)
    for w in weights:
        w.requires_grad_(True)
    _grad, _gtg = calculate_gradient(net, inputs, targets, weights)
    _Hg = autograd.grad(_gtg, weights)
    layer_cnt = 0
    grads = dict()
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            _wh = layer.weight.data
            _qhg = _wh * _Hg[layer_cnt]  # theta_q grad
            grads[old_modules[idx]] = torch.abs(_qhg)
            layer_cnt += 1
    # keep_masks, threshold = ranking_mask(grads, 0.02, verbose=2)
    # _r, _l_r_ls = get_keep_ratio(keep_masks, True)

    # 移除某一层
    # keep_masks[mask_key[0]] *= 0
    remove_layer = 14
    # keep_masks[mask_key[remove_layer]][0, :] *= 0  #
    # keep_masks[mask_key[remove_layer]][:, 0] *= 0  #
    # keep_masks[mask_key[remove_layer]][0, :, :, :] *= 0  # 移除某个过滤器
    keep_masks[mask_key[remove_layer]][:, 0, :, :] *= 0  # 移除某个通道
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight.data = copy_net_weights[old_modules[idx]] * keep_masks[old_modules[idx]]
    _grad, _gtg = calculate_gradient(net, inputs, targets)
    _Hg = autograd.grad(_gtg, weights)
    layer_cnt = 0
    grads_1 = dict()
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            _wh = layer.weight.data
            # _wh = copy_net_weights[old_modules[idx]]
            _qhg = _wh * _Hg[layer_cnt]  # theta_q grad
            grads_1[old_modules[idx]] = torch.abs(_qhg)
            layer_cnt += 1

    # 作图
    figure = PltScore(800, 400, 60)

    ax1 = figure.plt.subplot(1, 2, 1)

    # _g, _ind = torch.sort(torch.cat([torch.flatten(grads[mask_key[remove_layer]][0, :, :, :])]), descending=True)
    # np_g = _g.cpu().detach().numpy()
    # ax1.plot(range(1, len(np_g) + 1, 1), np_g, color=figure.color[0], label=f'过滤器的敏感度')
    # np_g = torch.flatten(grads_1[mask_key[remove_layer]][0, :, :, :])[_ind].cpu().detach().numpy()
    # ax1.scatter(range(1, len(np_g) + 1, 1), np_g, color=figure.color[3], s=1, label=f'移除过滤器')

    _g, _ind = torch.sort(torch.cat([torch.flatten(grads[mask_key[remove_layer]][:, 0, :, :])]), descending=True)
    np_g = _g.cpu().detach().numpy()
    ax1.plot(range(1, len(np_g) + 1, 1), np_g, color=figure.color[0], label=f'通道的敏感度')
    np_g = torch.flatten(grads_1[mask_key[remove_layer]][:, 0, :, :])[_ind].cpu().detach().numpy()
    ax1.scatter(range(1, len(np_g) + 1, 1), np_g, color=figure.color[3], s=1, label=f'通道置零之后')

    ax1.legend()

    # 观察上下层对应的梯度变化
    # ax2 = figure.plt.subplot(1, 2, 2)
    # _g, _ind = torch.sort(torch.cat([torch.flatten(grads[mask_key[remove_layer+1]][:, 0, :, :])]), descending=True)
    # np_g = _g.cpu().detach().numpy()
    # ax2.plot(range(1, len(np_g) + 1, 1), np_g, color=figure.color[0], label=f'下一层对应的通道')
    # np_g = torch.flatten(grads_1[mask_key[remove_layer+1]][:, 0, :, :])[_ind].cpu().detach().numpy()
    # ax2.scatter(range(1, len(np_g) + 1, 1), np_g, color=figure.color[3], s=1, label=f'移除过滤器之后')
    # ax2.legend()

    ax2 = figure.plt.subplot(1, 2, 2)
    _g, _ind = torch.sort(torch.cat([torch.flatten(grads[mask_key[remove_layer-1]][0, :, :, :])]), descending=True)
    np_g = _g.cpu().detach().numpy()
    ax2.plot(range(1, len(np_g) + 1, 1), np_g, color=figure.color[0], label=f'上一层对应的过滤器')
    np_g = torch.flatten(grads_1[mask_key[remove_layer-1]][0, :, :, :])[_ind].cpu().detach().numpy()
    ax2.scatter(range(1, len(np_g) + 1, 1), np_g, color=figure.color[3], s=1, label=f'通道置零之后')
    ax2.legend()

    figure.plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    figure.plt.show()

    return keep_masks, 0


# 画出无效权重
def Iterative_pruning_figure3(model, ratio, trainloader, device, config, num_iters):

    old_net = model
    old_modules = list(old_net.modules())
    keep_masks = reset_mask(old_net)
    last_masks = reset_mask(old_net)
    mask_key = [x for x in keep_masks.keys()]
    net = copy.deepcopy(model)
    net.train()
    copy_net_weights = dict()
    for layer in old_net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            copy_net_weights[layer] = torch.clone(layer.weight.data)

    data_mode = config.data_mode
    samples_per_class = config.samples_per_class
    num_classes = config.classe
    inputs, targets = fetch_data(trainloader, num_classes, samples_per_class, dm=data_mode)
    inputs = inputs.to(device)
    targets = targets.to(device)

    # 全网络的梯度和Hg
    weights = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weights.append(layer.weight)
    for w in weights:
        w.requires_grad_(True)
    _grad, _gtg = calculate_gradient(net, inputs, targets, weights)
    _Hg = autograd.grad(_gtg, weights)
    layer_cnt = 0
    grads = dict()
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            _wh = layer.weight.data
            _qhg = _wh * _Hg[layer_cnt]  # theta_q grad
            grads[old_modules[idx]] = torch.abs(_qhg)
            layer_cnt += 1
    keep_masks, threshold = ranking_mask(grads, 0.005, verbose=2)
    _r, _l_r_ls = get_keep_ratio(keep_masks, True)
    true_masks = effective_masks_synflow(model, keep_masks, (inputs, targets), device)
    _, _true_l_r = get_keep_ratio(true_masks, True)

    # 修剪98%的gtg
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight.data = copy_net_weights[old_modules[idx]] * keep_masks[old_modules[idx]]
    _grad_1, _gtg = calculate_gradient(net, inputs, targets)

    # 作图
    figure = PltScore(1000, 1000, 100)

    # fig_num = len(mask_key)
    fig_num = 4
    _row = math.ceil(fig_num**0.5)
    for i in range(fig_num):
        ax1 = figure.plt.subplot(_row, _row, i+1)
        _g, _ind = torch.sort(torch.cat([torch.flatten(_grad[i])]), descending=True)
        np_g = _g.cpu().detach().numpy()
        ax1.plot(range(1, len(np_g) + 1, 1), np_g, color=figure.color[0], label=f'全网络')

        np_g = torch.flatten(_grad_1[i])[_ind].cpu().detach().numpy()
        ax1.scatter(range(1, len(np_g) + 1, 1), np_g, color=figure.color[1], s=1, label=f'修剪，本层剩余{_l_r_ls[i]*100:.2f}%')
        non_effe = torch.flatten(keep_masks[mask_key[i]] - true_masks[mask_key[i]]).bool()
        np_g = torch.flatten(_grad_1[i])[non_effe].cpu().detach().numpy()
        ax1.scatter(np.arange(1, len(non_effe) + 1, 1)[non_effe.cpu().detach().numpy()], np_g, color=figure.color[3], s=2, label=f'无效权重，有效剩余{_true_l_r[i]*100:.2f}%')

        ax1.legend()

    figure.plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    figure.plt.show()

    return keep_masks, 0


# 对不同gtg的敏感测试
def Iterative_pruning_figure4(model, ratio, trainloader, device, config, num_iters):

    old_net = model
    old_modules = list(old_net.modules())
    keep_masks = reset_mask(old_net)
    last_masks = reset_mask(old_net)
    mask_key = [x for x in keep_masks.keys()]
    net = copy.deepcopy(model)
    net.train()
    copy_net_weights = dict()
    for layer in old_net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            copy_net_weights[layer] = torch.clone(layer.weight.data)

    data_mode = config.data_mode
    samples_per_class = config.samples_per_class
    num_classes = config.classe
    inputs, targets = fetch_data(trainloader, num_classes, samples_per_class, dm=data_mode)
    inputs = inputs.to(device)
    targets = targets.to(device)

    # 全网络的梯度和Hg
    weights = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weights.append(layer.weight)
    for w in weights:
        w.requires_grad_(True)
    _outputs = net.forward(inputs) / 200
    _loss = F.cross_entropy(_outputs, targets)
    _grad = autograd.grad(_loss, weights, create_graph=True)

    _gtg = []
    _layer = 0
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            _g = _grad[_layer]
            _gtg.append(_g.pow(2).sum())
            _layer += 1

    _Hg = []
    layer_num = len(mask_key)
    # == 层的gtg影响 ==
    for i in range(layer_num):
        _temp_gtg = 0
        # == 忽略前面层的gtg ==
        # for j in range(i, layer_num):
        #     _temp_gtg += _gtg[j]
        # == 邻层gtg ==
        _temp_gtg += _gtg[i]
        if i > 0:
            _temp_gtg += _gtg[i-1]
        elif i < layer_num-1:
            _temp_gtg += _gtg[i+1]
        _Hg.append(autograd.grad(_temp_gtg, weights[i], retain_graph=True)[0])

    # == Hg ==
    # _temp_gtg = 0
    # for i in range(layer_num):
    #     _temp_gtg += _gtg[i]
    # _Hg = autograd.grad(_temp_gtg, weights)

    layer_cnt = 0
    grads = dict()
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            _wh = layer.weight.data
            _qhg = _wh * _Hg[layer_cnt]  # theta_q grad
            grads[old_modules[idx]] = torch.abs(_qhg)
            layer_cnt += 1

    _grad, _gtg = calculate_gradient(net, inputs, targets, weights)
    _Hg = autograd.grad(_gtg, weights)
    layer_cnt = 0
    grads_1 = dict()
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            _wh = layer.weight.data
            _qhg = _wh * _Hg[layer_cnt]  # theta_q grad
            grads_1[old_modules[idx]] = torch.abs(_qhg)
            layer_cnt += 1

    keep_masks, threshold = ranking_mask(grads_1, 0.005, verbose=2)
    get_keep_ratio(keep_masks, True, verbose=1)
    keep_masks, threshold = ranking_mask(grads, 0.005, verbose=2)
    get_keep_ratio(keep_masks, True, verbose=1)

    # # 作图
    # figure = PltScore(1000, 1000, 100)
    #
    # # fig_num = len(mask_key)
    # fig_num = 4
    # _row = math.ceil(fig_num**0.5)
    # for i in range(fig_num):
    #     ax1 = figure.plt.subplot(_row, _row, i+1)
    #     _g, _ind = torch.sort(torch.cat([torch.flatten(grads[mask_key[i]])]), descending=True)
    #     np_g = _g.cpu().detach().numpy()
    #     ax1.plot(range(1, len(np_g) + 1, 1), np_g, color=figure.color[0], label=f'修改')
    #
    #     np_g = torch.flatten(grads_1[mask_key[i]])[_ind].cpu().detach().numpy()
    #     ax1.scatter(range(1, len(np_g) + 1, 1), np_g, color=figure.color[1], s=1, label=f'')
    #
    #     ax1.legend()
    #
    # figure.plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    # figure.plt.show()

    return keep_masks, 0


# 迭代可视化
def Iterative_pruning_figure5(model, ratio, trainloader, device, config, num_iters):

    old_net = model
    old_modules = list(old_net.modules())
    keep_masks = reset_mask(old_net)
    net = copy.deepcopy(model)
    net.train()
    copy_net_weights = dict()
    for layer in old_net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            copy_net_weights[layer] = torch.clone(layer.weight.data)

    inputs, targets = fetch_data(trainloader, config.classe, config.samples_per_class, dm=config.data_mode)
    inputs = inputs.to(device)
    targets = targets.to(device)

    num_iters = config.num_iters_prune
    desc = ('[keep ratio=%s] acceptable score: %e' % (1, 0))
    prog_bar = tqdm(range(num_iters), total=num_iters, desc=desc, leave=True) if num_iters > 1 else range(num_iters)
    for epoch in prog_bar:
        keep_ratio = (1.0 - ratio) ** ((epoch + 1) / num_iters)

        net.zero_grad()
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight.data = copy_net_weights[old_modules[idx]] * keep_masks[old_modules[idx]]

        score_mode = config.score_mode
        gradg_list = hessian_gradient_product(net, (inputs, targets), device, config)

        layer_cnt = 0
        grads = dict()
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                kxt = 0
                _w = copy_net_weights[old_modules[idx]] if config.dynamic else layer.weight.data
                if score_mode == 1:
                    for i in range(len(gradg_list)):
                        _qhg = _w * gradg_list[i][layer_cnt]  # theta_q grad
                        kxt += _qhg
                elif score_mode == 2:
                    for i in range(len(gradg_list)):
                        _qhg = _w * gradg_list[i][layer_cnt]  # theta_q grad
                        kxt += _qhg
                    kxt = torch.abs(kxt)
                elif score_mode == 3:
                    kxt = 1e6
                    for i in range(len(gradg_list)):
                        _qhg = _w * gradg_list[i][layer_cnt]  # theta_q grad
                        kxt *= torch.abs(_qhg)
                elif score_mode == 4:
                    aef = 1e6
                    for i in range(len(gradg_list)):
                        _qhg = _w * gradg_list[i][layer_cnt] * aef  # theta_q grad
                        kxt += _qhg.pow(2)
                    kxt = kxt.sqrt()
                if isinstance(layer, nn.Conv2d):
                    if 'filter' in config.debug:
                        # (n, c, k, k)
                        _s = kxt.shape
                        kxt = torch.mean(kxt, dim=(1, 2, 3), keepdim=True).repeat(1, _s[1], _s[2], _s[3])
                    elif 'channel' in config.debug:
                        _s = kxt.shape
                        kxt = torch.mean(kxt, dim=(0, 2, 3), keepdim=True).repeat(_s[0], 1, _s[2], _s[3])
                grads[old_modules[idx]] = kxt

                # # 层敏感
                # temp = torch.mean(torch.abs(kxt)) - torch.abs(torch.mean(kxt))
                # grads[old_modules[idx]] = kxt/temp

                layer_cnt += 1

        oir_mask = None if config.dynamic else keep_masks
        keep_masks, threshold = ranking_mask(grads, keep_ratio, verbose=2, oir_mask=oir_mask)
        if 'a' in config.debug:
            _score = kernel_link_score(grads, keep_masks)
            keep_masks, threshold = ranking_mask(_score, keep_ratio, verbose=2)

        # keep_masks, threshold = Single_ranking_pruning(model, 1-keep_ratio, ([inputs], [targets]), device, config, reinit=False, retrun_inf=0, verbose=2)

        # keep_masks, _ = SynFlow(model, 1-keep_ratio, (inputs, targets), device, 1, ori_masks=keep_masks)

        if num_iters > 1:
            desc = ('[keep ratio=%s]' % keep_ratio)
            prog_bar.set_description(desc, refresh=True)

        # # 作图
        # if keep_ratio < 0.05:
        # # if keep_ratio < 0.026 and keep_ratio > 0.015:
        #     figure = PltScore(500, 500, 60)
        #
        #     mask_key = [x for x in keep_masks.keys()]
        #     fig_num = len(mask_key)-1
        #     # fig_num = 4
        #     _row = math.ceil(fig_num**0.5)
        #     for i, (m, k) in enumerate(keep_masks.items()):
        #         if i > -1:
        #         # if i > 11:
        #             if isinstance(m, nn.Conv2d):
        #                 # (n, c, k, k)
        #                 _2d = torch.sum(k, dim=(2, 3))
        #                 # _2d = (torch.sum(k, dim=(2, 3)) > 0).float()
        #                 np_2d = _2d.cpu().detach().numpy()
        #                 figure.plt.subplot(_row, _row, i + 1)
        #                 # figure.plt.subplot(_row, _row, i + 1 - 12)
        #                 figure.plt.imshow(np_2d, cmap=figure.plt.cm.hot)
        #                 if i == 0:
        #                     figure.plt.text(-30, 0, f'{keep_ratio*100:0.2f}%')
        #                 # figure.plt.colorbar()
        #             elif isinstance(m, nn.Linear):
        #                 pass
        #
        #     figure.plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        #     # figure.plt.show()
        #     # _path = f'runs/plt_out/iter_dyn1/it{epoch}_{keep_ratio*100:0.2f}.jpg'
        #     # _path = f'runs/plt_out/iter_sta0/it{epoch}_{keep_ratio*100:0.2f}.jpg'
        #     # _path = f'runs/plt_out/iter_syn1/it{epoch}_{keep_ratio*100:0.2f}.jpg'
        #     _path = f'runs/plt_out/iter_snip/it{epoch}_{keep_ratio*100:0.2f}.jpg'
        #     figure.plt.savefig(_path)
        #     figure.plt.close()

    return keep_masks, 0


# 敏感保留的 不敏感移除的
def Iterative_pruning_figure6(model, ratio, trainloader, device, config, num_iters):

    old_net = model
    old_modules = list(old_net.modules())
    keep_masks = reset_mask(old_net)
    net = copy.deepcopy(model)
    net.train()
    copy_net_weights = dict()
    for layer in old_net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            copy_net_weights[layer] = torch.clone(layer.weight.data)

    inputs, targets = fetch_data(trainloader, config.classe, config.samples_per_class, dm=config.data_mode)
    inputs = inputs.to(device)
    targets = targets.to(device)

    num_iters = config.num_iters_prune
    desc = ('[keep ratio=%s] acceptable score: %e' % (1, 0))
    prog_bar = tqdm(range(num_iters), total=num_iters, desc=desc, leave=True) if num_iters > 1 else range(num_iters)
    for epoch in prog_bar:
        keep_ratio = (1.0 - ratio) ** ((epoch + 1) / num_iters)

        net.zero_grad()
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight.data = copy_net_weights[old_modules[idx]] * keep_masks[old_modules[idx]]

        weights = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                weights.append(layer.weight)
        for w in weights:
            w.requires_grad_(True)
        _outputs = net.forward(inputs) / 200
        _loss = F.cross_entropy(_outputs, targets)
        _grad = autograd.grad(_loss, weights, create_graph=True)

        # == 测试重合度 静态保持在0.8左右 动态飘动 ==
        _gtg = 0
        keep_gtg, remove_gtg = 0, 0
        _layer = 0
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                _g = _grad[_layer]
                _gtg += _g.pow(2).sum()

                keep_gtg += (_g*keep_masks[old_modules[idx]]).pow(2).sum()
                remove_gtg += (_g*(1-keep_masks[old_modules[idx]])).pow(2).sum()

                _layer += 1

        oir_mask = None if config.dynamic else keep_masks
        grads = dict()
        _Hg = autograd.grad(_gtg, weights, retain_graph=True)
        layer_cnt = 0
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                _wh = copy_net_weights[old_modules[idx]] if config.dynamic else layer.weight.data
                _qhg = _wh * _Hg[layer_cnt]  # theta_q grad
                grads[old_modules[idx]] = torch.abs(_qhg)
                layer_cnt += 1
        keep_masks, threshold = ranking_mask(grads, keep_ratio, verbose=2, oir_mask=oir_mask)
        del grads

        k_grads = dict()
        _Hg = autograd.grad(keep_gtg, weights, retain_graph=True)
        layer_cnt = 0
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                _wh = copy_net_weights[old_modules[idx]] if config.dynamic else layer.weight.data
                _qhg = _wh * _Hg[layer_cnt]  # theta_q grad
                k_grads[old_modules[idx]] = torch.abs(_qhg)
                layer_cnt += 1
        k_masks, threshold = ranking_mask(k_grads, keep_ratio, verbose=2, oir_mask=oir_mask)

        r_grads = dict()
        _Hg = autograd.grad(remove_gtg, weights, retain_graph=True)
        layer_cnt = 0
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                _wh = copy_net_weights[old_modules[idx]] if config.dynamic else layer.weight.data
                _qhg = _wh * _Hg[layer_cnt]  # theta_q grad
                r_grads[old_modules[idx]] = -torch.abs(_qhg)
                layer_cnt += 1
        r_masks, threshold = ranking_mask(r_grads, keep_ratio, verbose=2, oir_mask=oir_mask)

        _coin = coincide_mask(k_masks, r_masks)  # 重合度
        print(_coin)

        # == 敏感于保留且不敏感于移除 ==
        # keep_gtg, remove_gtg = 0, 0
        # _layer = 0
        # for idx, layer in enumerate(net.modules()):
        #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #         _g = _grad[_layer]
        #         keep_gtg += (_g*keep_masks[old_modules[idx]]).pow(2).sum()
        #         remove_gtg += (_g*(1-keep_masks[old_modules[idx]])).pow(2).sum()
        #
        #         _layer += 1
        #
        # oir_mask = None if config.dynamic else keep_masks
        # # grads = dict()
        # k_grads = dict()
        # r_grads = dict()
        # keep_Hg = autograd.grad(keep_gtg, weights, retain_graph=True)
        # remove_Hg = autograd.grad(remove_gtg, weights, retain_graph=True)
        # layer_cnt = 0
        # for idx, layer in enumerate(net.modules()):
        #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #         _wh = copy_net_weights[old_modules[idx]] if config.dynamic else layer.weight.data
        #         k_qhg = _wh * keep_Hg[layer_cnt]
        #         r_qhg = _wh * remove_Hg[layer_cnt]
        #         # grads[old_modules[idx]] = torch.abs(k_qhg) / torch.abs(r_qhg) if epoch > 0 else torch.abs(k_qhg)
        #         k_grads[old_modules[idx]] = torch.abs(k_qhg)
        #         r_grads[old_modules[idx]] = -torch.abs(r_qhg)
        #         layer_cnt += 1
        # # k_masks, threshold = ranking_mask(k_grads, keep_ratio, verbose=2, oir_mask=oir_mask)
        # # r_masks, threshold = ranking_mask(r_grads, keep_ratio, verbose=2, oir_mask=oir_mask)
        # keep_masks, threshold = ranking_mask(r_grads, keep_ratio, verbose=2, oir_mask=oir_mask)
        #
        # # _coin = coincide_mask(k_masks, r_masks)

        if num_iters > 1:
            # desc = ('[keep ratio=%s]' % keep_ratio)
            desc = ('[keep ratio=%s] acceptable score: %e' % (keep_ratio, threshold))
            prog_bar.set_description(desc, refresh=True)

        # 作图
        # figure = PltScore(1000, 1000, 100)
        # mask_key = [x for x in keep_masks.keys()]
        # # fig_num = len(mask_key)
        # fig_num = 4
        # _row = math.ceil(fig_num ** 0.5)
        # for i in range(fig_num):
        #     # ax1 = figure.plt.subplot(_row, _row, i + 1)
        #     # _g, _ind = torch.sort(torch.cat([torch.flatten(k_grads[mask_key[i]])]), descending=True)
        #     # np_g = _g.cpu().detach().numpy()
        #     # ax1.plot(range(1, len(np_g) + 1, 1), np_g, color=figure.color[0], label=f'')
        #     # np_g = torch.flatten(r_grads[mask_key[i]])[_ind].cpu().detach().numpy()
        #     # ax1.scatter(range(1, len(np_g) + 1, 1), np_g, color=figure.color[1], s=1, label=f'')
        #     # ax1.legend()
        #
        #     ax1 = figure.plt.subplot(_row, _row, i + 1)
        #     _g, _ind = torch.sort(torch.cat([torch.flatten(grads[mask_key[i]])]), descending=True)
        #     np_g = _g.cpu().detach().numpy()
        #     ax1.plot(range(1, len(np_g) + 1, 1), np_g, color=figure.color[0], label=f'')
        #
        # figure.plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        # figure.plt.show()

    return keep_masks, 0


# 层敏感比较
def Iterative_pruning_figure7(model, ratio, trainloader, device, config, num_iters):

    old_net = model
    old_modules = list(old_net.modules())
    keep_masks = reset_mask(old_net)
    net = copy.deepcopy(model)
    net.train()
    copy_net_weights = dict()
    for layer in old_net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            copy_net_weights[layer] = torch.clone(layer.weight.data)

    inputs, targets = fetch_data(trainloader, config.classe, config.samples_per_class, dm=config.data_mode)
    inputs = inputs.to(device)
    targets = targets.to(device)

    iter_layer_score = []
    num_iters = config.num_iters_prune
    desc = ('[keep ratio=%s] acceptable score: %e' % (1, 0))
    prog_bar = tqdm(range(num_iters), total=num_iters, desc=desc, leave=True) if num_iters > 1 else range(num_iters)
    for epoch in prog_bar:
        keep_ratio = (1.0 - ratio) ** ((epoch + 1) / num_iters)

        net.zero_grad()
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight.data = copy_net_weights[old_modules[idx]] * keep_masks[old_modules[idx]]

        score_mode = config.score_mode
        gradg_list = hessian_gradient_product(net, (inputs, targets), device, config)

        layer_cnt = 0
        grads = dict()
        layer_score = []
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                kxt = 0
                _w = copy_net_weights[old_modules[idx]] if config.dynamic else layer.weight.data
                if score_mode == 1:
                    for i in range(len(gradg_list)):
                        _qhg = _w * gradg_list[i][layer_cnt]  # theta_q grad
                        kxt += _qhg
                elif score_mode == 2:
                    for i in range(len(gradg_list)):
                        _qhg = _w * gradg_list[i][layer_cnt]  # theta_q grad
                        kxt += _qhg
                    kxt = torch.abs(kxt)
                elif score_mode == 3:
                    kxt = 1e6
                    for i in range(len(gradg_list)):
                        _qhg = _w * gradg_list[i][layer_cnt]  # theta_q grad
                        kxt *= torch.abs(_qhg)
                elif score_mode == 4:
                    aef = 1e6
                    for i in range(len(gradg_list)):
                        _qhg = _w * gradg_list[i][layer_cnt] * aef  # theta_q grad
                        kxt += _qhg.pow(2)
                    kxt = kxt.sqrt()
                if isinstance(layer, nn.Conv2d):
                    if 'filter' in config.debug:
                        # (n, c, k, k)
                        _s = kxt.shape
                        kxt = torch.mean(kxt, dim=(1, 2, 3), keepdim=True).repeat(1, _s[1], _s[2], _s[3])
                    elif 'channel' in config.debug:
                        _s = kxt.shape
                        kxt = torch.mean(kxt, dim=(0, 2, 3), keepdim=True).repeat(_s[0], 1, _s[2], _s[3])
                grads[old_modules[idx]] = kxt
                layer_score.append(torch.mean(kxt).cpu().detach().numpy())
                layer_cnt += 1

        oir_mask = None if config.dynamic else keep_masks
        keep_masks, threshold = ranking_mask(grads, keep_ratio, verbose=2, oir_mask=oir_mask)
        if 'a' in config.debug:
            _score = kernel_link_score(grads, keep_masks)
            keep_masks, threshold = ranking_mask(_score, keep_ratio, verbose=2)

        # keep_masks, threshold = Single_ranking_pruning(model, 1-keep_ratio, ([inputs], [targets]), device, config, reinit=False, retrun_inf=0, verbose=2)

        # keep_masks, _ = SynFlow(model, 1-keep_ratio, (inputs, targets), device, 1, ori_masks=keep_masks)

        if num_iters > 1:
            desc = ('[keep ratio=%s]' % keep_ratio)
            prog_bar.set_description(desc, refresh=True)

        iter_layer_score.append(layer_score)

    # 作图
    figure = PltScore(500, 500, 60)

    for i in range(len(iter_layer_score[0])):
        figure.plt.plot(range(len(iter_layer_score)), [iter_layer_score[j][i] for j in range(len(iter_layer_score))], color=figure.color[i%10], linestyle='--', marker='o', label=f'{i}')
        figure.plt.legend()

    figure.plt.yscale('log')
    figure.plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    figure.plt.show()

    return keep_masks, 0


# 层敏感变化
def Iterative_pruning_figure8(model, ratio, trainloader, device, config, num_iters):

    old_net = model
    old_modules = list(old_net.modules())
    keep_masks = reset_mask(old_net)
    net = copy.deepcopy(model)
    net.train()
    copy_net_weights = dict()
    for layer in old_net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            copy_net_weights[layer] = torch.clone(layer.weight.data)

    inputs, targets = fetch_data(trainloader, config.classe, config.samples_per_class, dm=config.data_mode)
    inputs = inputs.to(device)
    targets = targets.to(device)

    iter_layer_score = []
    last_score = []
    num_iters = config.num_iters_prune
    desc = ('[keep ratio=%s] acceptable score: %e' % (1, 0))
    prog_bar = tqdm(range(num_iters), total=num_iters, desc=desc, leave=True) if num_iters > 1 else range(num_iters)
    for epoch in prog_bar:
        keep_ratio = (1.0 - ratio) ** ((epoch + 1) / num_iters)

        net.zero_grad()
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight.data = copy_net_weights[old_modules[idx]] * keep_masks[old_modules[idx]]

        score_mode = config.score_mode
        gradg_list = hessian_gradient_product(net, (inputs, targets), device, config)

        layer_cnt = 0
        grads = dict()
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                kxt = 0
                _w = copy_net_weights[old_modules[idx]] if config.dynamic else layer.weight.data
                if score_mode == 1:
                    for i in range(len(gradg_list)):
                        _qhg = _w * gradg_list[i][layer_cnt]  # theta_q grad
                        kxt += _qhg
                elif score_mode == 2:
                    for i in range(len(gradg_list)):
                        _qhg = _w * gradg_list[i][layer_cnt]  # theta_q grad
                        kxt += _qhg
                    kxt = torch.abs(kxt)
                elif score_mode == 3:
                    kxt = 1e6
                    for i in range(len(gradg_list)):
                        _qhg = _w * gradg_list[i][layer_cnt]  # theta_q grad
                        kxt *= torch.abs(_qhg)
                elif score_mode == 4:
                    aef = 1e6
                    for i in range(len(gradg_list)):
                        _qhg = _w * gradg_list[i][layer_cnt] * aef  # theta_q grad
                        kxt += _qhg.pow(2)
                    kxt = kxt.sqrt()
                if isinstance(layer, nn.Conv2d):
                    if 'filter' in config.debug:
                        # (n, c, k, k)
                        _s = kxt.shape
                        kxt = torch.mean(kxt, dim=(1, 2, 3), keepdim=True).repeat(1, _s[1], _s[2], _s[3])
                    elif 'channel' in config.debug:
                        _s = kxt.shape
                        kxt = torch.mean(kxt, dim=(0, 2, 3), keepdim=True).repeat(_s[0], 1, _s[2], _s[3])
                grads[old_modules[idx]] = kxt
                layer_cnt += 1

        oir_mask = None if config.dynamic else keep_masks
        keep_masks, threshold = ranking_mask(grads, keep_ratio, verbose=2, oir_mask=oir_mask)
        if 'a' in config.debug:
            _score = kernel_link_score(grads, keep_masks)
            keep_masks, threshold = ranking_mask(_score, keep_ratio, verbose=2)

        layer_score = []
        for i, (m, g) in enumerate(grads.items()):
            # _index = (oir_mask[m]-keep_masks[m]).bool()  # 将要修剪的权重
            _index = (keep_masks[m]).bool()  # 保留的权重
            temp_s = torch.mean(torch.abs(g[_index])).cpu().detach().numpy()
            layer_score.append(temp_s)

            # temp_s = torch.mean(torch.abs(g)).cpu().detach().numpy()  # 层敏感
            # if epoch == 0:
            #     last_score.append(temp_s)
            # else:
            #     layer_score.append(temp_s - last_score[i])
            #     last_score[i] = temp_s
        if epoch > -1:
            iter_layer_score.append(layer_score)

        # keep_masks, threshold = Single_ranking_pruning(model, 1-keep_ratio, ([inputs], [targets]), device, config, reinit=False, retrun_inf=0, verbose=2)

        # keep_masks, _ = SynFlow(model, 1-keep_ratio, (inputs, targets), device, 1, ori_masks=keep_masks)

        if num_iters > 1:
            desc = ('[keep ratio=%s]' % keep_ratio)
            prog_bar.set_description(desc, refresh=True)


    # 作图
    figure = PltScore(500, 500, 60)

    for i in range(len(iter_layer_score[0])):
        figure.plt.plot(range(len(iter_layer_score)), [iter_layer_score[j][i] for j in range(len(iter_layer_score))], color=figure.color[figure.cs[i+7]], linestyle='--', marker='o', label=f'{i}')
        figure.plt.legend()

    # figure.plt.yscale('log')
    figure.plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
    figure.plt.show()

    return keep_masks, 0


# 梯度范数作图
def Iterative_pruning_figure9(model, ratio, trainloader, device, config, num_iters):

    old_net = model
    old_modules = list(old_net.modules())
    keep_masks = reset_mask(old_net)
    net = copy.deepcopy(model)
    net.train()
    copy_net_weights = dict()
    for layer in old_net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            copy_net_weights[layer] = torch.clone(layer.weight.data)

    inputs, targets = fetch_data(trainloader, config.classe, config.samples_per_class, dm=config.data_mode)
    inputs = inputs.to(device)
    targets = targets.to(device)

    # 作图
    figure = PltScore(500, 500, 60)

    iter_layer_score = []
    pruning_ratio = []
    grad_mode = config.grad_mode
    num_iters = config.num_iters_prune
    desc = ('[keep ratio=%s] acceptable score: %e' % (1, 0))
    prog_bar = tqdm(range(num_iters), total=num_iters, desc=desc, leave=True) if num_iters > 1 else range(num_iters)
    for epoch in prog_bar:
        keep_ratio = (1.0 - ratio) ** ((epoch + 1) / num_iters)

        # net.zero_grad()
        # for idx, layer in enumerate(net.modules()):
        #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #         # layer.weight.data = copy_net_weights[old_modules[idx]] * keep_masks[old_modules[idx]]
        #         layer.weight.data = copy_net_weights[old_modules[idx]]  # 恢复
        #
        # weights = []
        # for layer in net.modules():
        #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #         weights.append(layer.weight)
        # for w in weights:
        #     w.requires_grad_(True)
        # _outputs = net.forward(inputs) / 200
        # _loss = F.cross_entropy(_outputs, targets)
        # _grad = autograd.grad(_loss, weights, create_graph=True)
        #
        # # == gtg ==
        # _gtg = 0
        # keep_gtg, remove_gtg = 0, 0
        # _layer = 0
        # for idx, layer in enumerate(net.modules()):
        #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #         _g = _grad[_layer]
        #         _gtg += _g.pow(2).sum()
        #         # keep_gtg += (_g*keep_masks[old_modules[idx]]).pow(2).sum()
        #         # remove_gtg += (_g*(1-keep_masks[old_modules[idx]])).pow(2).sum()
        #         _layer += 1
        #
        # oir_mask = None if config.dynamic else keep_masks
        # grads = dict()
        # if grad_mode == 0:
        #     _Hg = autograd.grad(_gtg, weights, retain_graph=True)
        # layer_cnt = 0
        # for idx, layer in enumerate(net.modules()):
        #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #         _wh = copy_net_weights[old_modules[idx]] if config.dynamic else layer.weight.data
        #         if grad_mode == 0:
        #             _qhg = _wh * _Hg[layer_cnt]  # theta_q grad
        #             grads[old_modules[idx]] = _qhg
        #         elif grad_mode == 3:
        #             _qhg = _wh * _grad[layer_cnt]  # theta_q grad
        #             grads[old_modules[idx]] = torch.abs(_qhg)
        #         layer_cnt += 1
        # keep_masks, threshold = ranking_mask(grads, keep_ratio, verbose=2, oir_mask=oir_mask)

        keep_masks, threshold = Single_ranking_pruning(model, 1-keep_ratio, ([inputs], [targets]), device, config, reinit=False, retrun_inf=0, verbose=2)

        # keep_masks, _ = SynFlow(model, 1-keep_ratio, (inputs, targets), device, 1, ori_masks=keep_masks)

        if num_iters > 1:
            desc = ('[keep ratio=%s]' % keep_ratio)
            prog_bar.set_description(desc, refresh=True)

        # 计算修剪之后网络的梯度
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight.data = copy_net_weights[old_modules[idx]] * keep_masks[old_modules[idx]]
        weights = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                weights.append(layer.weight)
        for w in weights:
            w.requires_grad_(True)
        _outputs = net.forward(inputs)
        _loss = F.cross_entropy(_outputs, targets)
        _grad = autograd.grad(_loss, weights, create_graph=True)

        # layer_score = []
        # for i, (m, g) in enumerate(grads.items()):
        #     temp_s = _grad[i].pow(2).sum().cpu().detach().numpy()
        #     # _index = (keep_masks[m]).bool()  # 保留的权重
        #     # temp_s = torch.sum(_grad[i].pow(2)[_index]).cpu().detach().numpy()
        #     layer_score.append(temp_s)
        # iter_layer_score.append(layer_score)
        # pruning_ratio.append(1-keep_ratio)

        # 作图
        # for l, (m, s) in enumerate(grads.items()):
        for l, g in enumerate(_grad):
            if l == 1:
                # _s, index = torch.sort(torch.cat([torch.flatten(g.pow(2))]), descending=True)
                _s, index = torch.sort(torch.cat([torch.flatten(torch.abs(g))]), descending=True)
                # _s, index = torch.sort(torch.cat([torch.flatten(s)]), descending=True)
                np_s = _s.cpu().detach().numpy()
                figure.plt.plot(range(1, len(np_s) + 1, 1), np_s, color=figure.color[figure.cs[epoch + 7]], label=f'{(1 - keep_ratio) * 100:0.2f}%')
                # figure.plt.axvline(np.argwhere(np_s > min(np_s))[-1], color=figure.color[figure.cs[epoch + 7]], linestyle=':')


    # 作图
    # figure = PltScore(500, 500, 60)
    #
    # for i in range(len(iter_layer_score[0])):
    #     figure.plt.plot(range(len(iter_layer_score)),
    #                     [iter_layer_score[j][i] for j in range(len(iter_layer_score))],
    #                     color=figure.color[figure.cs[i + 7]], linestyle='--', marker='o', label=f'{i}')
    #     figure.plt.legend()
    #
    # # figure.plt.yscale('log')
    # scale_x = [f'{x*100:0.2f}%' for x in pruning_ratio]
    # figure.plt.gca().set_xticks(range(len(iter_layer_score)))
    # figure.plt.gca().set_xticklabels(scale_x)
    # figure.plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
    # figure.plt.show()

    figure.plt.xscale('log')
    figure.plt.axhline(0, color='gray', linestyle=':')
    figure.plt.legend(loc='upper right')
    figure.plt.show()

    return keep_masks, 0


def normalization(scores):
    all_scores = torch.cat([torch.flatten(x) for x in scores.values()])
    mean_div = all_scores.mean()
    all_scores.div_(mean_div)
    std = all_scores.std()
    mean = all_scores.mean()
    # print(mean, std)
    for m, s in scores.items():
        scores[m] = (scores[m]/mean_div - mean) / std
    return scores


def Force_choose(model, dataloader, device, config):

    @torch.no_grad()
    def linearize(model):
        signs = {}
        for name, param in model.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(model, signs):
        for name, param in model.state_dict().items():
            param.mul_(signs[name])

    old_net = model
    net = copy.deepcopy(model)
    net.eval()
    net.zero_grad()
    modules_ls = list(old_net.modules())
    signs = linearize(net)
    (data, _) = next(iter(dataloader))
    input_dim = list(data[0, :].shape)
    input = torch.ones([1] + input_dim).to(device)
    copy_net_weights = dict()
    for layer in old_net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            copy_net_weights[layer] = torch.clone(layer.weight.data)
    keep_masks = reset_mask(old_net)
    score = dict()

    inputs, targets = fetch_data(dataloader, config.classe, config.samples_per_class, dm=config.data_mode)
    inputs = inputs.to(device)
    targets = targets.to(device)

    ratio = config.target_ratio
    num_iters = config.num_iters_prune
    desc = ('[keep ratio=%s] acceptable score: %e' % (1, 0))
    prog_bar = tqdm(range(num_iters), total=num_iters, desc=desc, leave=True) if num_iters > 1 else range(num_iters)
    for epoch in prog_bar:
        keep_ratio = (1.0 - ratio)**((epoch + 1) / num_iters)
        # for idx, layer in enumerate(net.modules()):
        #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #         layer.weight.data = (copy_net_weights[modules_ls[idx]] * keep_masks[modules_ls[idx]]).abs_()
        # net.zero_grad()
        # # forward
        # output = net(input)
        # torch.sum(output).backward()
        # # score
        # rank_score = Single_ranking_pruning(old_net, ratio, dataloader, device, config, retrun_inf=9, verbose=0)
        # for idx, layer in enumerate(net.modules()):
        #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #         _w = copy_net_weights[modules_ls[idx]] if config.dynamic else layer.weight.data
        #         score[modules_ls[idx]] = (_w * layer.weight.grad).abs_()*rank_score[modules_ls[idx]]

        net.train()
        net.zero_grad()
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight.data = (copy_net_weights[modules_ls[idx]] * keep_masks[modules_ls[idx]])

        # === Calculate SNIP and GraSP (g, Hg) ===
        weights = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                weights.append(layer.weight)
        for w in weights:
            w.requires_grad_(True)
        _outputs = net.forward(inputs) / 200
        _loss = F.cross_entropy(_outputs, targets)
        _grad = autograd.grad(_loss, weights, create_graph=True)
        _gtg, _layer = 0, 0
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                _g = _grad[_layer]
                _gtg += _g.pow(2).sum()
                _layer += 1
        _Hg = autograd.grad(_gtg, weights, retain_graph=True)

        # === Calculate SynFlow (g) ===
        net.eval()
        net.zero_grad()
        signs = linearize(net)
        _output = net(input)
        torch.sum(_output).backward()
        nonlinearize(net, signs)

        # === Calculate score ===
        score = dict()
        s_synflow = dict()
        # s_snip = dict()
        s_grasp = dict()
        layer_cnt = 0
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                _w = copy_net_weights[modules_ls[idx]]
                _synflow = _w * layer.weight.grad
                # _snip = _w * _grad[layer_cnt]
                _grasp = _w * _Hg[layer_cnt]
                s_synflow[modules_ls[idx]] = _synflow.abs_()
                # s_snip[modules_ls[idx]] = _snip.abs_()
                s_grasp[modules_ls[idx]] = _grasp.abs_()
                layer_cnt += 1
        s_synflow = normalization(s_synflow)
        # s_snip = normalization(s_snip)
        s_grasp = normalization(s_grasp)
        for m, s in s_synflow.items():
            score[m] = 1*s_synflow[m]+0.9*s_grasp[m]
        keep_masks = dict()
        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([torch.flatten(x) for x in score.values()])
        threshold, _index = torch.topk(all_scores, int(len(all_scores) * keep_ratio))
        acceptable_score = threshold[-1]
        for m, g in score.items():
            keep_masks[m] = (g >= acceptable_score).float()

        # masks
        # keep_masks, acceptable_score = ranking_mask(score, keep_ratio, False, verbose=2)

        if num_iters > 1:
            desc = ('[keep ratio=%s] acceptable score: %e' % (keep_ratio, acceptable_score))
            prog_bar.set_description(desc, refresh=True)

    # nonlinearize(net, signs)

    return keep_masks, score


def masks_compare(mb, masks, trainloader, device, inf_str=''):
    pr_str = '-'*20
    pr_str += '\n%s' % inf_str if len(inf_str) > 0 else ''
    mb.register_mask(masks)
    mb_ratios = mb.get_ratio_at_each_layer()
    pr_str += '\n** %s - Remaining: %.5f%%' % ('', mb_ratios['ratio'])
    true_masks = effective_masks_synflow(mb.model, masks, trainloader, device)
    mb.register_mask(true_masks)
    effective_ratios = mb.get_ratio_at_each_layer()
    pr_str += '\n** %s - Remaining: %.5f%%\n' % ('true_masks', effective_ratios['ratio'])
    pr_str += '-'*20
    print(pr_str)
    return pr_str, mb_ratios['ratio'], effective_ratios['ratio']


def print_mask_information(mb, logger=None, inf_str=''):
    ratios = mb.get_ratio_at_each_layer()
    if logger:
        logger.info('** %s - Mask information of %s. Overall Remaining: %.5f%%' % (inf_str, mb.get_name(), ratios['ratio']))
    re_str = '** %s - Mask information of %s. Overall Remaining: %.5f%%\n' % (inf_str, mb.get_name(), ratios['ratio'])
    count = 0
    for k, v in ratios.items():
        if k == 'ratio':
            continue
        if logger:
            logger.info('  (%d) %s: Remaining: %.5f%%' % (count, k, v))
        re_str += '  (%d) %.5f%%\n' % (count, v)
        count += 1
    return re_str


def kernel_link_score(score, masks):
    _channel_score = dict()
    _filter_score = dict()
    for m, s in score.items():
        _num_w = torch.sum((masks[m] == 1))
        if isinstance(m, nn.Conv2d):
            # (n, c, k, k)
            _channel = torch.sum(s*masks[m], dim=(0, 2, 3))/_num_w
            _filter = torch.sum(s*masks[m], dim=(1, 2, 3))/_num_w
        elif isinstance(m, nn.Linear):
            # (c, n)
            _channel = torch.sum(s*masks[m], dim=0)/_num_w
            _filter = torch.sum(s*masks[m], dim=1)/_num_w
        _channel_score[m] = _channel
        _filter_score[m] = _filter

    score_key = [x for x in score.keys()]
    layer_num = len(score_key)
    kl_mean = 0
    for i, (m, s) in enumerate(score.items()):
        # print(f'layer:{i+1}')
        # if i == 0 or i == layer_num-1:
        #     pass
        shape = s.shape
        if i == 0:
            _rear = _channel_score[score_key[i + 1]][:, None, None, None].repeat(1, shape[1], shape[2], shape[3])
            _front = _rear
        elif i == layer_num-1:
            _front = _filter_score[score_key[i-1]][None, :].repeat(shape[0], 1)
            _rear = _front
        else:
            # shape = s.shape
            if len(shape) == 4:
                _front = _filter_score[score_key[i-1]][None, :, None, None].repeat(shape[0], 1, shape[2], shape[3])
                _rear = _channel_score[score_key[i+1]][:, None, None, None].repeat(1, shape[1], shape[2], shape[3])
            elif len(shape) == 2:
                _front = _filter_score[score_key[i-1]][None, :].repeat(shape[0], 1)
                _rear = _channel_score[score_key[i+1]][:, None].repeat(1, shape[1])
            else:
                NotImplementedError('Layer shape unsupported ' + shape)
            kernel_link = _front+_rear
            # kl_mean += torch.mean(kernel_link)
            score[m] += kernel_link
    # kl_mean /= (layer_num-2)
    # score[score_key[0]] += kl_mean
    # score[score_key[layer_num-1]] += kl_mean

    return score


def layer_balanced_score(score):
    _layer_score = dict()
    for m, s in score.items():
        _layer_score[m] = torch.mean(s).abs_() * 1e10

    score_key = [x for x in score.keys()]
    layer_num = len(score_key)
    for i, (m, s) in enumerate(score.items()):
        for j in range(layer_num):
            if i != j:
                score[m] *= _layer_score[score_key[j]]

    return score


def get_model_hg(hg_ls, masks, score_mode):
    keep_hg = 0
    remove_hg = 0
    for l, (m, s) in enumerate(masks.items()):
        kxt = 0
        if score_mode == 1:
            for i in range(len(hg_ls)):
                kxt += hg_ls[i][l]
        elif score_mode == 2:
            for i in range(len(hg_ls)):
                kxt += hg_ls[i][l]
            kxt = torch.abs(kxt)
        elif score_mode == 3:
            kxt = 1e6  # 约等于超参，估计值，kxt是👴
            for i in range(len(hg_ls)):
                kxt *= torch.abs(hg_ls[i][l])
        elif score_mode == 4:
            aef = 1e6  # 约等于超参，估计值
            for i in range(len(hg_ls)):
                kxt += (hg_ls[i][l] * aef).pow(2)
            kxt = kxt.sqrt()
        keep_hg += torch.sum(kxt*s)
        remove_hg += torch.sum(kxt*(1-s))
    return keep_hg.cpu().detach().numpy(), remove_hg.cpu().detach().numpy()


class PltScore(object):

    def __init__(self, xpixels=500, ypixels=500, dpi=None):
        import matplotlib.pyplot as plt
        plt.rcParams['axes.unicode_minus'] = False  # -
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rc('font', family='Times New Roman')
        if dpi is None:
            plt.figure(1)
        else:
            xinch = xpixels / dpi
            yinch = ypixels / dpi
            plt.figure(figsize=(xinch, yinch))
        self.plt = plt
        # self.color = ['red', 'orange', 'yellow', 'green', 'blue', 'cyan', 'purple', 'pink', 'coral', 'gold', 'lime', 'navy', 'teal', 'indigo']
        import matplotlib.colors as mcolors
        self.cs = list(mcolors.CSS4_COLORS.keys())
        self.color = mcolors.CSS4_COLORS


    def plt_score(self, score, layer=1, cnt=0, label=''):
        for l, (m, s) in enumerate(score.items()):
            if l == layer:
                _s, _ind = torch.sort(torch.cat([torch.flatten(s)]), descending=True)
                np_s = _s.cpu().detach().numpy()
                self.plt.plot(range(1, len(np_s) + 1, 1), np_s, color=self.color[cnt], label=label)
                # self.plt.axvline(np.argwhere(np_s > min(np_s))[-1], color=self.color[iter], linestyle=':')

    def plt_end(self):
        self.plt.xscale('log')
        self.plt.axhline(0, color='gray', linestyle=':')
        self.plt.legend(loc='upper right')
        self.plt.show()


def Pruner(mb, trainloader, device, config):
    net = mb.model
    """
        config.prune_mode: 
            dense, rank, rank/random, rank/iterative, coin
    """
    if 'dense' in config.prune_mode:
        masks, _ = Single_ranking_pruning(net, 0, None, None, config)
    elif 'rank' in config.prune_mode:
        if config.rank_algo.lower() == 'synflow':
            masks, _ = SynFlow(net, config.target_ratio, trainloader, device, config.num_iters_prune, dynamic=config.dynamic)
        else:
            if 'iterative' in config.prune_mode:
                # masks, _ = Iterative_pruning(net, config.target_ratio, trainloader, device, config)
                # masks, _ = Iterative_pruning_figure(net, config.target_ratio, trainloader, device, config, config.num_iters_prune)
                masks, _ = Iterative_pruning_figure9(net, config.target_ratio, trainloader, device, config, config.num_iters_prune)
            else:
                if 'oir' in config.debug:
                    masks, _score = Single_ranking_pruning(net, config.target_ratio, trainloader, device, config, reinit=True, retrun_inf=1)
                else:
                    masks, _score = Single_ranking_pruning2(net, config.target_ratio, trainloader, device, config, reinit=True, retrun_inf=1)  # 最大不同组 最小同组

                # if config.debug == 'd':
                #     plot = PltScore()
                #     plot.plt_score(_score, cnt=0, label='rank')
                #     _score = layer_balanced_score(_score)
                #     plot.plt_score(_score, cnt=1, label='balance')
                #     plot.plt_end()
                #     masks = ranking_mask(_score, 1-config.target_ratio, verbose=1)
                # if config.debug == 'a':
                #     masks, _score = Single_ranking_pruning(net, config.target_ratio, trainloader, device, config, reinit=True, retrun_inf=1)
                #     _score = kernel_link_score(_score, masks)
                #     masks, threshold = ranking_mask(_score, 1-config.target_ratio, verbose=2)
        if 'random' in config.prune_mode:
            for m, g in masks.items():
                shape = g.shape
                perm = torch.randperm(g.nelement())
                masks[m] = g.reshape(-1)[perm].reshape(shape)
            get_connected_scores(masks, 'rank/random', 1)

    elif 'force' == config.prune_mode:
        masks, _ = Force_choose(net, trainloader, device, config)
    else:
        raise NotImplementedError('Prune mode unsupported ' + config.prune_mode)

    return masks, 0
