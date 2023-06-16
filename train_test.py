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
from tqdm import tqdm
from utils.common_utils import PresetLRScheduler


def train(net, loader, optimizer, criterion, lr_scheduler, epoch, writer, lr_mode):
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

    prog_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

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


def test(net, loader, criterion, epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    # prog_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            # prog_bar.set_description(desc, refresh=True)

    acc = 100. * correct / total
    print('test[epoch{}]-----loss:{:.8f},test_acc:{:.2f}%'.format(epoch,test_loss/total,acc))

    # writer.add_scalar('test/loss', test_loss / (batch_idx + 1), epoch)
    # writer.add_scalar('test/acc', 100. * correct / total, epoch)
    return test_loss,acc


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
            train(net, trainloader, optimizer, criterion, lr_scheduler, epoch, writer, lr_mode)
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

