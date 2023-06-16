# -*- coding: utf-8 -*-

"""
Created on 03/23/2022
train_test.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from tqdm import tqdm
from utils.common_utils import PresetLRScheduler
from pruner.pruning_exp import fetch_data
import copy


def writer_exp(net, masks, loader, criterion, writer, epoch, num_classes, gtg_mode, ignore_grad, exp_data, exp_name, train_mode=False):
    if train_mode:
        net.train()
    else:
        net.eval()
    # 重新计算损失和建图，便于求一阶二阶导
    weights = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weights.append(layer.weight)
    for w in weights:
        w.requires_grad_(True)
    # layer_key
    layer_key = [x for x in masks.keys()]

    # 10个类别，每类样本取10个
    samples_per_class = 10
    if num_classes == 100:
        samples_per_class = 2
    if gtg_mode == 0:
        inputs, targets = fetch_data(loader, num_classes, samples_per_class, mode=0, dm=0)  # 不同标签分组 label
    elif gtg_mode == 1:
        inputs, targets = fetch_data(loader, num_classes, samples_per_class, mode=0, dm=1)  # 同标签分组 data
    elif gtg_mode == 2:
        inputs, targets = fetch_data(loader, 1, num_classes, 1)  # 随机取100个样本 random
    else:
        inputs, targets = exp_data
    inputs = inputs.cuda()
    targets = targets.cuda()

    # 直接分两个组
    num_group = 2
    N = inputs.shape[0]
    equal_parts = N // num_group
    grad_ls = []
    gss_ls = []
    for i in range(num_group):
        _outputs = net.forward(inputs[i * equal_parts:(i + 1) * equal_parts])
        _loss = criterion(_outputs, targets[i * equal_parts:(i + 1) * equal_parts])
        grad_ls.append(autograd.grad(_loss, weights, create_graph=True))

    for i in range(num_group):
        for j in range(i + 1, num_group):
            _gz = 0
            _gz_hi = 0
            _gz_hj = 0
            _layer = 0
            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    if ignore_grad == 0:
                        _gz += (grad_ls[i][_layer] * grad_ls[j][_layer]).sum()  # g1 * g2
                    else:
                        _gz += (grad_ls[i][_layer] * grad_ls[j][_layer] * masks[layer_key[_layer]]).sum()  # g1 * g2
                    _layer += 1
            gss_ls.append(_gz)

    # D1 + D2
    _outputs = net.forward(inputs)
    _loss = criterion(_outputs, targets)
    _g = autograd.grad(_loss, weights, create_graph=True)
    gtg = 0
    _layer = 0
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if ignore_grad == 0:
                gtg += _g[_layer].pow(2).sum()  # g * g
            else:
                gtg += (_g[_layer] * masks[layer_key[_layer]]).pow(2).sum()  # g * g
            _layer += 1

    # ------------- debug ----------------
    # use_layer = 2
    # print('all_g', torch.mean(_g[use_layer]), torch.sum(_g[use_layer]))
    # print('g1', torch.mean(grad_ls[0][use_layer]), torch.sum(grad_ls[0][use_layer]))
    # print('g2', torch.mean(grad_ls[1][use_layer]), torch.sum(grad_ls[1][use_layer]))

    # tensorboard
    # print(exp_name, gss_ls[0])
    if train_mode:
        exp_name = 'tra_' + exp_name
    writer.add_scalar(exp_name + '/gtg', gtg, epoch)
    writer.add_scalar(exp_name + '/gss', gss_ls[0], epoch)


def loss_coupling(net, loader, criterion, scheduler, epoch, writer, samples_num=256):

    net = copy.deepcopy(net).train()
    net.zero_grad()
    _optim = optim.SGD(net.parameters(), lr=scheduler.get_last_lr()[0], momentum=0.9, weight_decay=0.0005)

    inputs, targets = fetch_data(loader, 1, samples_num, 1)  # 随机取样本
    inputs = inputs.cuda()
    targets = targets.cuda()

    # 直接分两个组，di dj
    num_group = 2
    N = inputs.shape[0]
    equal_parts = N // num_group
    # 计算 di 训练前 di dj 的损失值
    i = 0
    _outputs = net.forward(inputs[i * equal_parts:(i + 1) * equal_parts])
    before_loss_i = criterion(_outputs, targets[i * equal_parts:(i + 1) * equal_parts])
    i = 1
    _outputs = net.forward(inputs[i * equal_parts:(i + 1) * equal_parts])
    before_loss_j = criterion(_outputs, targets[i * equal_parts:(i + 1) * equal_parts])
    # 使用 di 训练
    i = 0
    _optim.zero_grad()
    _outputs = net.forward(inputs[i * equal_parts:(i + 1) * equal_parts])
    _loss = criterion(_outputs, targets[i * equal_parts:(i + 1) * equal_parts])
    _loss.backward()
    _optim.step()
    # 计算 di 训练后 di dj 的损失值
    i = 0
    _outputs = net.forward(inputs[i * equal_parts:(i + 1) * equal_parts])
    after_loss_i = criterion(_outputs, targets[i * equal_parts:(i + 1) * equal_parts])
    i = 1
    _outputs = net.forward(inputs[i * equal_parts:(i + 1) * equal_parts])
    after_loss_j = criterion(_outputs, targets[i * equal_parts:(i + 1) * equal_parts])

    writer.add_scalar('train/delta_loss_%d' % samples_num, before_loss_i-after_loss_i, epoch)
    writer.add_scalar('train/coupling_loss_%d' % samples_num, before_loss_j-after_loss_j, epoch)


def train(net, masks, loader, optimizer, criterion, lr_scheduler, epoch, writer, lr_mode, debug='0', classe=10):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if lr_mode == 'cosine':
        _lr = lr_scheduler.get_last_lr()
    elif 'preset' in lr_mode:
        _lr = lr_scheduler.get_lr(optimizer)
    desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (_lr, 0, 0, correct, total))
    writer.add_scalar('train/lr', _lr, epoch)

    # === 前两个batch的gigj ===
    # if 'writer' in debug:
    #     gi = dict()

    prog_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        if batch_idx == 0:  # 真实损失 一阶二阶近似的差距
            loss.backward(create_graph=True, retain_graph=True)
            if 'writer' in debug:
                _gtg = 0
                for idx, layer in enumerate(net.modules()):
                    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                        _gtg += (layer.weight.grad).pow(2).sum()
                epsilon = _lr[0]
                first_order = -1 * epsilon * _gtg

                # ----------计算 second_order----------
                # 重新计算损失和建图，便于求一阶二阶导
                weights = []
                for layer in net.modules():
                    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                        weights.append(layer.weight)
                for w in weights:
                    w.requires_grad_(True)

                _gHg = 0
                Hg = autograd.grad(_gtg, weights, retain_graph=True)
                _cnt = 0
                for idx, layer in enumerate(net.modules()):
                    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                        _gHg += (layer.weight.grad * Hg[_cnt]).sum()
                        _cnt += 1
                second_order = first_order + 0.5 * epsilon * epsilon * _gHg
        else:
            loss.backward()
        optimizer.step()

        # === 前两个batch的gigj ===
        # if 'writer' in debug:
        #     if batch_idx == 0:
        #         for idx, layer in enumerate(net.modules()):
        #             if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #                 gi[layer] = copy.deepcopy(layer.weight.grad)
        #     if batch_idx == 1:
        #         g1g2 = 0
        #         g1g1 = 0
        #         g2g2 = 0
        #         for idx, layer in enumerate(net.modules()):
        #             if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #                 g1g2 += (layer.weight.grad * gi[layer]).sum()  # gi * gj
        #                 g1g1 += (gi[layer]).pow(2).sum()
        #                 g2g2 += (layer.weight.grad).pow(2).sum()
        #         # print('train/g1g2', g1g2)
        #         # print('train/g1g1', g1g1)
        #         # print('train/g2g2', g2g2)
        #         writer.add_scalar('train/g1g2', g1g2, epoch)
        #         writer.add_scalar('train/g1g1', g1g1, epoch)
        #         writer.add_scalar('train/g2g2', g2g2, epoch)
        # === 第一个batch的gigj ===
        # if 'writer' in debug:
        #     if batch_idx == 0:
        #         _gtg = 0
        #         for idx, layer in enumerate(net.modules()):
        #             if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #                 _gtg += (layer.weight.grad).pow(2).sum()
        #
        #         # ----------计算 gigj----------
        #         # 重新计算损失和建图，便于求一阶二阶导
        #         weights = []
        #         for layer in net.modules():
        #             if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #                 weights.append(layer.weight)
        #         for w in weights:
        #             w.requires_grad_(True)
        #         # layer_key
        #         layer_key = [x for x in masks.keys()]
        #
        #         # 按标签分两个组
        #         _index = targets < (classe/2)
        #         # print(targets[_index])
        #         # print(targets[~_index])
        #         _outputs = net.forward(inputs[_index])
        #         _loss = criterion(_outputs, targets[_index])
        #         _gi = autograd.grad(_loss, weights, retain_graph=True)
        #         _outputs = net.forward(inputs[~_index])
        #         _loss = criterion(_outputs, targets[~_index])
        #         _gj = autograd.grad(_loss, weights, retain_graph=True)
        #         _gigj = 0
        #         _layer = 0
        #         for layer in net.modules():
        #             if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #                 _gigj += (_gi[_layer] * _gj[_layer] * masks[layer_key[_layer]]).sum()  # gi * gj
        #                 _layer += 1
        #         # --------------------
        #
        #         # print('train/gtg', _gtg)
        #         # print('train/gigj', _gigj)
        #         writer.add_scalar('train/gtg', _gtg, epoch)
        #         writer.add_scalar('train/gigj', _gigj, epoch)
        # === 第一个batch的真实损失和一阶二阶项 ===
        if 'writer' in debug:
            if batch_idx == 0:
                loss_after = criterion(net(inputs), targets)
                delta_loss = loss_after - loss

                # print(delta_loss)
                # print(first_order)
                # print(second_order)
                writer.add_scalar('train/delta_loss', delta_loss, epoch)
                writer.add_scalar('train/first_order', first_order, epoch)
                writer.add_scalar('train/second_order', second_order, epoch)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if lr_mode == 'cosine':
            _lr = lr_scheduler.get_last_lr()
        elif 'preset' in lr_mode:
            lr_scheduler(optimizer, epoch)
            _lr = lr_scheduler.get_lr(optimizer)
        desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (_lr, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    writer.add_scalar('train/loss', train_loss / (batch_idx + 1), epoch)
    writer.add_scalar('train/acc', 100. * correct / total, epoch)


def test(net, loader, criterion, epoch, writer):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss / (0 + 1), 0, correct, total))

    prog_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

    acc = 100. * correct / total

    writer.add_scalar('test/loss', test_loss / (batch_idx + 1), epoch)
    writer.add_scalar('test/acc', 100. * correct / total, epoch)
    return acc


def train_once(mb, trainloader, testloader, config, writer, logger, pretrain=None, lr_mode='cosine', optim_mode='SGD'):
    net = mb.model
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    num_epochs = config.epoch
    criterion = nn.CrossEntropyLoss()
    if optim_mode == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if lr_mode == 'cosine':
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.001)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif 'preset' in lr_mode:
        lr_schedule = {0: learning_rate,
                       int(num_epochs * 0.5): learning_rate * 0.1,
                       int(num_epochs * 0.75): learning_rate * 0.01}
        lr_scheduler = PresetLRScheduler(lr_schedule)
    else:
        print('===!!!=== Wrong learning rate decay setting! ===!!!===')
        exit()

    print_inf = ''
    best_epoch = 0
    if pretrain:
        best_acc = pretrain['acc']
        continue_epoch = pretrain['epoch']
    else:
        best_acc = 0
        continue_epoch = -1
    for epoch in range(num_epochs):
        if epoch > continue_epoch:  # 其他时间电表空转
            # if 'writer' in config.debug:
            #     # loss_coupling(net, trainloader, criterion, lr_scheduler, epoch, writer, 64)
            #     # loss_coupling(net, trainloader, criterion, lr_scheduler, epoch, writer, 128)
            #     # loss_coupling(net, trainloader, criterion, lr_scheduler, epoch, writer, 512)
            #     writer_exp(net, mb.masks, trainloader, criterion, writer, epoch, config.classe, 0, 1, None, 'label')
            #     writer_exp(net, mb.masks, trainloader, criterion, writer, epoch, config.classe, 1, 1, None, 'data')
            #     writer_exp(net, mb.masks, trainloader, criterion, writer, epoch, 128, 2, 1, None, 'random')
            #     # writer_exp(net, mb.masks, trainloader, criterion, writer, epoch, config.classe, 0, 1, None, 'label', True)
            #     # writer_exp(net, mb.masks, trainloader, criterion, writer, epoch, config.classe, 1, 1, None, 'data', True)
            #     # writer_exp(net, mb.masks, trainloader, criterion, writer, epoch, 128, 2, 1, None, 'random', True)
            train(net, mb.masks, trainloader, optimizer, criterion, lr_scheduler, epoch, writer, lr_mode, debug=config.debug, classe=config.classe)
            test_acc = test(net, testloader, criterion, epoch, writer)

            if test_acc > best_acc and epoch > 10:
                print('Saving..')
                state = {
                    'net': net,
                    'acc': test_acc,
                    'epoch': epoch,
                    'args': config,
                    'mask': mb.masks,
                    'ratio': mb.get_ratio_at_each_layer()
                }
                path = os.path.join(config.checkpoint_dir, 'train_%s_best.pth.tar' % config.exp_name)
                torch.save(state, path)
                best_acc = test_acc
                best_epoch = epoch
        if lr_mode == 'cosine':
            lr_scheduler.step()
        else:
            lr_scheduler(optimizer, epoch)

    logger.info('best acc: %.4f, epoch: %d' %
                (best_acc, best_epoch))
    return 'best acc: %.4f, epoch: %d\n' % (best_acc, best_epoch), print_inf

