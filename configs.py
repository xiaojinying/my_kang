# -*- coding: utf-8 -*-

"""
Created on 03/23/2022
configs.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

import os
import time
import argparse
from utils import mail_log
from easydict import EasyDict as edict
import logging
from tensorboardX import SummaryWriter


def init_config():
    parser = argparse.ArgumentParser()
    ## Training Hyperparameters ##
    parser.add_argument('--config', type=str, default='cifar10/vgg19/98',
                        help='config (dataset/model(network depth)/pruning rate(percentage)) - e.g.[cifar10/vgg19/98, mnist/lenet5/90]')
    parser.add_argument('--pretrained', type=str, default='666',
                        help='path - runs/anoi/finetune_cifar10_vgg19_l2_best.pth.tar')
    parser.add_argument('--run', type=str, default='test_exp', help='experimental notes (default: test_exp)')
    parser.add_argument('--epoch', type=int, default=666)
    parser.add_argument('--batch_size', type=int, default=666)
    parser.add_argument('--l2', type=str, default=666)
    parser.add_argument('--lr_mode', type=str, default='cosine', help='cosine or preset')
    parser.add_argument('--optim_mode', type=str, default='SGD', help='SGD or Adam')
    parser.add_argument('--storage_mask', type=int, default=0)  # 存储mask
    parser.add_argument('--debug', type=str, default='p')  # 调试标记（打印数据和作图等）
    parser.add_argument('--dp', type=str, default='../Data', help='dataset path')
    ## Pruning Hyperparameters ##
    parser.add_argument('--rank_algo', type=str, default='grasp',
                        help='rank algorithm (choose one of: snip, grasp, synflow, gcs, gcs-group)')
    parser.add_argument('--prune_mode', type=str, default='rank/random',
                        help='prune mode (choose one of: dense, rank, rank/random, rank/iterative)')
    parser.add_argument('--train_mode', type=int, default=1)
    parser.add_argument('--data_mode', type=int, default=666)  # 数据模式
    parser.add_argument('--grad_mode', type=int, default=666)
    parser.add_argument('--score_mode', type=int, default=666)
    parser.add_argument('--num_group', type=int, default=666)  # 分组数
    parser.add_argument('--samples_per', type=int, default=666)
    parser.add_argument('--num_iters', type=int, default=666)
    parser.add_argument('--num_iters_prune', type=int, default=100)
    parser.add_argument('--dynamic', type=int, default=1)
    args = parser.parse_args()

    ## Default Config ##
    exp_set = args.config.split('/')  # dataset model prune_ratio
    exp_name = f'{exp_set[0]}_{exp_set[1]}_prune{exp_set[2]}'.replace('.', '_')
    base_config = {'network': "vgg", 'depth': 19, 'dataset': 'cifar10',
                   'batch_size': 128, 'epoch': 180, 'learning_rate': 0.1, 'weight_decay': 5e-4,
                   'target_ratio': 0.90, 'samples_per_class': 10}
    config = edict(base_config)
    config.dataset = exp_set[0]
    config.target_ratio = float(exp_set[2])/100.0
    config.network = ''.join(list(filter(str.isalpha, exp_set[1])))
    config.depth = int(''.join(list(filter(str.isdigit, exp_set[1]))))
    if 'mnist' in config.dataset:
        config.batch_size = 256
        config.epoch = 80
        config.weight_decay = 1e-4
        config.classe = 10
    elif 'cifar' in config.dataset:
        config.batch_size = 128
        config.epoch = 180
        config.weight_decay = 5e-4
        if config.dataset == 'cifar10':
            config.classe = 10
        else:
            config.classe = 100
            if 'resnet' in config.network:
                config.samples_per_class = 5
                config.num_iters = 2
    elif 'imagenet' in config.dataset:
        config.batch_size = 128
        config.epoch = 300
        config.classe = 200
        if 'vgg' in config.network:
            config.weight_decay = 5e-4
            config.samples_per_class = 5
            config.num_iters = 2
        elif 'resnet' in config.network:
            config.weight_decay = 1e-4
            config.samples_per_class = 1
            config.num_iters = 10
    # if args.rank_algo.lower() == 'synflow':
    config.num_iters_prune = 100

    ## Experiment Name and Out Path ##
    summn = [exp_name]
    chekn = [exp_name]
    if len(args.run) > 0:
        summn.append(args.run)
        chekn.append(args.run)
        config.exp_name = exp_name + '_' + args.run
    _str = ''.join(list(filter(str.isalpha, args.prune_mode)))
    summn[-1] += f'_{_str}'
    chekn[-1] += f'_{_str}'
    config.exp_name += f'_{_str}'
    if args.rank_algo != '666':
        summn[-1] += f'_{args.rank_algo}'
        chekn[-1] += f'_{args.rank_algo}'
        config.exp_name += f'_{args.rank_algo}'
    summn.append("summary/")
    chekn.append("checkpoint/")
    summary_dir = ["./runs/pruning"] + exp_set[:-1] + summn
    ckpt_dir = ["./runs/pruning"] + exp_set[:-1] + chekn
    config.summary_dir = os.path.join(*summary_dir)
    config.checkpoint_dir = os.path.join(*ckpt_dir)
    print("=> config.summary_dir:    %s" % config.summary_dir)
    print("=> config.checkpoint_dir: %s" % config.checkpoint_dir)

    ## Console Parameters ##
    if args.pretrained != '666':
        config.pretrained = args.pretrained
        print("use pre-trained mode:{}".format(config.pretrained))
    else:
        config.pretrained = None
    if args.epoch != 666:
        config.epoch = args.epoch
        print("set new epoch:{}".format(config.epoch))
    if args.batch_size != 666:
        config.batch_size = args.batch_size
        print("set new batch_size:{}".format(config.batch_size))
    if args.l2 != 666:
        config.weight_decay = args.l2
        print("set new weight_decay:{}".format(config.weight_decay))
    config.prune_mode = args.prune_mode
    config.train_mode = args.train_mode
    config.storage_mask = args.storage_mask
    config.lr_mode = args.lr_mode
    config.optim_mode = args.optim_mode
    config.debug = args.debug
    config.dp = args.dp
    config.dynamic = args.dynamic
    config.send_mail_head = (config.exp_name + ' -> ' + args.run + '\n')
    config.send_mail_str = (mail_log.get_words() + '\n')
    config.send_mail_str += "=> 我能在河边钓一整天的🐟 <=\n"
    if args.rank_algo != '666':
        # gcs: --data_mode 1 --grad_mode 2 --score_mode 2 --num_group 2
        # gcs-group: --data_mode 1 --grad_mode 2 --score_mode 4 --num_group 5
        # grasp: --data_mode 0 --grad_mode 0 --score_mode 1 --num_group 1
        # snip: --data_mode 0 --grad_mode 3 --score_mode 2 --num_group 1
        if args.rank_algo.lower() == 'gcs':
            config.data_mode = 1
            config.grad_mode = 2
            config.score_mode = 2
            config.num_group = 2
        elif args.rank_algo.lower() == 'gcs-group':
            config.data_mode = 1
            config.grad_mode = 2
            config.score_mode = 4
            config.num_group = 5
        elif args.rank_algo.lower() == 'gcs-max':
            config.data_mode = 1
            config.grad_mode = 4
            config.score_mode = 4
            config.num_group = 5
        elif args.rank_algo.lower() == 'grasp':
            config.data_mode = 0
            config.grad_mode = 0
            config.score_mode = 1
            config.num_group = 1
        elif args.rank_algo.lower() == 'grass':
            config.data_mode = 0
            config.grad_mode = 0
            config.score_mode = 2
            config.num_group = 1
        elif args.rank_algo.lower() == 'snip':
            config.data_mode = 0
            config.grad_mode = 3
            config.score_mode = 2
            config.num_group = 1
        elif args.rank_algo.lower() == 'synflow':
            pass
        else:
            print("choose one of: GCS, GCS-Group, GraSP, SNIP, SynFlow")
            config.prune_mode = 'dense'  # 不剪枝
        config.rank_algo = args.rank_algo
        print("set pruning algorithm: {}".format(args.rank_algo))
    if args.samples_per != 666:
        config.samples_per_class = args.samples_per
        print("set new samples_per_class:{}".format(config.samples_per_class))
    if args.num_iters != 666:
        config.num_iters = args.num_iters
        print("set new num_iters:{}".format(config.num_iters))
    if args.num_group != 666:
        if 10 % args.num_group == 0:
            config.num_group = args.num_group
        else:
            config.num_group = 2
            print("num_group must be divisible by the number of tags")
    if args.data_mode != 666:
        config.data_mode = args.data_mode
    if args.grad_mode != 666:
        config.grad_mode = args.grad_mode
    if args.score_mode != 666:
        config.score_mode = args.score_mode
    if args.num_iters_prune != 666:
        config.num_iters_prune = args.num_iters_prune

    # if 'exp' in config.exp_name:
    #     config.epoch = 0

    return config


def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


def get_logger(name, logpath, displaying=True, saving=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    log_path = logpath + name + time.strftime("-%Y%m%d-%H%M%S")
    makedirs(log_path)
    if saving:
        info_file_handler = logging.FileHandler(log_path, encoding='utf-8')
        info_file_handler.setLevel(logging.INFO)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger


def init_logger(config):
    makedirs(config.summary_dir)
    makedirs(config.checkpoint_dir)

    # set logger
    logger = get_logger('log', logpath=config.summary_dir + '/')
    logger.info(dict(config))
    writer = SummaryWriter(config.summary_dir)
    return logger, writer