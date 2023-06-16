# -*- coding: utf-8 -*-

"""
Created on 04/03/2022
main_gan.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""
import torch

from models.model_base import ModelBase
from models.base.init_utils import weights_init
from configs import *
from utils.network_utils import get_network
from utils.data_utils import get_dataloader
from pruner.pruning import *
from train_test import *


def main():
    config = init_config()
    logger, writer = init_logger(config)


    model = get_network(config.network, config.depth, config.dataset, use_bn=config.get('use_bn', True))
    mb = ModelBase(config.network, config.depth, config.dataset, model)
    mb.cuda()

    # ===== get dataloader =====
    trainloader, testloader = get_dataloader(config.dataset, config.batch_size, 256, 4, root=config.dp)
    # ===== pruning =====
    mb.model.apply(weights_init)
    masks = reset_mask(mb.model)
    mask_key = [x for x in masks.keys()]
    wat = copy.deepcopy(mb.model)
    for layer in wat.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight.data = torch.ones_like(layer.weight.data)

    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(wat.parameters(), lr=0.1)

    keep_ratio = 1-config.target_ratio
    masks, _score = Single_ranking_pruning(mb.model, config.target_ratio, trainloader, 'cuda', config, reinit=True, retrun_inf=1)
    # 乘作用力
    layer_cnt = 0
    for layer in wat.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # _score[mask_key[layer_cnt]] *= layer.weight.data
            masks[mask_key[layer_cnt]] = _score[mask_key[layer_cnt]] * layer.weight.data
            layer_cnt += 1
    # # 对应剪枝率下的阈值
    # all_scores = torch.cat([torch.flatten(x) for x in _score.values()])
    # k = int(keep_ratio * all_scores.numel())
    # _topk, _index = torch.topk(all_scores, k)
    # threshold = _topk[-1]
    # print(f'threshold: {threshold}')
    # # 得到掩码
    # for m, s in masks.items():
    #     # masks[m] = torch.relu(torch.tanh(_score[m]-threshold))
    #     masks[m] = ((_score[m]) >= threshold).float()
    true_masks = effective_masks_synflow(mb.model, masks, trainloader, 'cuda')
    optimizer.zero_grad()
    force_loss = sum([mse_loss(masks[m], true_masks[m]) for m in masks.keys()])
    masks_compare(mb, masks, trainloader, 'cuda')
    print(f'force_loss: {force_loss}, keep_ratio: {get_keep_ratio(masks)}, effective_ratio: {get_keep_ratio(true_masks)}')
    force_loss.backward()
    optimizer.step()

    #
    mb.register_mask(masks)
    print_inf = print_mask_information(mb, logger)
    config.send_mail_str += print_inf
    pr_str, rem_ratio, eff_ratio = masks_compare(mb, masks, trainloader, 'cuda')
    config.send_mail_str += pr_str
    logger.info('  LR: %.5f, WD: %.5f, Epochs: %d' % (config.learning_rate, config.weight_decay, config.epoch))
    config.send_mail_str += 'LR: %.5f, WD: %.5f, Epochs: %d, Batch: %d \n' % (config.learning_rate, config.weight_decay, config.epoch, config.batch_size)
    if abs((1-config.target_ratio)*100 - rem_ratio) >= 1:
        print("ERROR: Pruning ratio not as expected")
        quit()
    if eff_ratio == 0 or eff_ratio < rem_ratio*0.05:  # 有效压缩率过低，不训练
        print("ERROR: Effective compression ratio is too low")
        quit()
    # ===== train =====
    tr_str, print_inf = train_once(mb, trainloader, testloader, config, writer, logger, None, config.lr_mode, config.optim_mode)
    config.send_mail_str += print_inf
    config.send_mail_str += tr_str
    if 'test' not in config.exp_name:
        QQmail = mail_log.MailLogs()
        QQmail.sendmail(config.send_mail_str, header=config.send_mail_head)


if __name__ == '__main__':
    main()

