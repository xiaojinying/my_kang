# -*- coding: utf-8 -*-

"""
Created on 03/23/2022
main.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

from models.model_base import ModelBase
from models.base.init_utils import weights_init
from configs import *
from utils.network_utils import get_network
from utils.data_utils import get_dataloader
from pruner.pruning import *
from train import *


def main():
    print('rand_seed:', torch.initial_seed())
    config = init_config()
    logger, writer = init_logger(config)

    state = None
    # ===== build/load model =====
    if config.pretrained:
        state = torch.load(config.pretrained)
        model = state['net']
        masks = state['mask']
        config.send_mail_str += f"use pre-trained mode -> acc:{state['acc']} epoch:{state['epoch']}\n"
        config.network = state['args'].network
        config.depth = state['args'].depth
        config.dataset = state['args'].dataset
        config.batch_size = state['args'].batch_size
        config.learning_rate = state['args'].learning_rate
        config.weight_decay = state['args'].weight_decay
        config.epoch = state['args'].epoch
        config.target_ratio = state['args'].target_ratio
        print('load model finish')
        print(state['args'])
    else:
        model = get_network(config.network, config.depth, config.dataset, use_bn=config.get('use_bn', True))
        masks = None

    mb = ModelBase(config.network, config.depth, config.dataset, model)
    mb.cuda()

    # ===== get dataloader =====
    trainloader, testloader = get_dataloader(config.dataset, config.batch_size, 256, 4, root=config.dp)
    # ===== pruning =====
    if masks is None:
        logger.info('** Target ratio: %.5f' % (config.target_ratio))
        mb.model.apply(weights_init)
        print("=> Applying weight initialization(%s)." % config.get('init_method', 'kaiming'))
        masks, _ = Pruner(mb, trainloader, 'cuda', config)
    mb.register_mask(masks)
    print_mask_information(mb, logger)
    pr_str, rem_ratio, eff_ratio = masks_compare(mb, masks, trainloader, 'cuda')
    print_mask_information(mb, logger)
    if abs((1-config.target_ratio)*100 - rem_ratio) >= 1:
        _str = "ERROR: Pruning ratio not as expected\n"
        print(_str)
        config.send_mail_str += _str
        config.epoch = 5
        # quit()
    if eff_ratio == 0 or eff_ratio < rem_ratio*0.05:  # 有效压缩率过低，不训练
        _str = "ERROR: Effective compression ratio is too low\n"
        print(_str)
        config.send_mail_str += _str
        config.epoch = 5
        # quit()

    train_eval_loop_1(mb, trainloader, testloader, config)


if __name__ == '__main__':
    main()
