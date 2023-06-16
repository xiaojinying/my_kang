# -- coding: utf-8 --
import numpy as np

from models.model_base import ModelBase
from models.base.init_utils import weights_init
from configs import *
from utils.network_utils import get_network
from utils.data_utils import get_dataloader
from pruner.pruning import *
from train_test import *
from copy import deepcopy

def train_eval_loop(mb, trainloader, testloader, config):
    ori_loss=[]
    ori_acc=[]
    net = mb.model
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    num_epochs = config.epoch
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    for epoch in range(num_epochs):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        lr = lr_scheduler.get_last_lr()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
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
            _lr = lr_scheduler.get_last_lr()
            if(batch_idx%100==0):
                print('train[epoch:{}]-----loss:{:.4f},acc:{:.2f}%,lr:{:.8f}'.format(epoch,train_loss/total,correct/total*100,_lr))
        lr_scheduler.step()
        loss,acc=test(net, testloader, criterion, epoch)
        ori_loss.append(loss)
        ori_acc.append(acc)
    np.save('ori_loss.npy', ori_loss)
    np.save('ori_acc.npy', ori_acc)

def train_eval_loop_1(mb, trainloader, testloader, config):
    # 对theta r和theta p进行更新，得到新模型，然后再进行梯度加权
    # model.load_state_dict(torch.load('125.pth'))
    net = mb.model
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    num_epochs = config.epoch
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    beta=-1
    loss2=nn.MSELoss()
    loss_test = []
    acc_test = []
    mask = []
    prune_mask = []
    for i in net.modules():
        if isinstance(i, nn.Conv2d) or isinstance(i, nn.Linear):
            mask.append(torch.clone(mb.masks[i]).detach())
            prune_mask.append(torch.clone(-1 * (mb.masks[i] - 1)).detach())
    for epoch in range(int(num_epochs)):
        net.train()
        mb.apply_masks()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(trainloader):
            net.zero_grad()
            model2 = deepcopy(net).cuda()
            mb.unregister_mask(model2)
            data, target = data.cuda(), target.cuda()
            op=optim.SGD(net.parameters(), lr=optimizer.state_dict()['param_groups'][0]['lr'], momentum=0.9, weight_decay=4e-4)

            output = model2(data)
            loss = criterion(output, target)
            loss.backward()
            op.step()
            net.zero_grad()
            model2.zero_grad()
            f1=F.softmax(net(data),dim=1)
            f2=F.softmax(model2(data),dim=1)
            l=loss2(f1,f2)
            l.backward()
            k=0
            gr2=[]
            for i in net.modules():
                if isinstance(i, nn.Conv2d) or isinstance(i, nn.Linear):
                    gr2.append(torch.clone(i.weight.grad).detach())
                    k+=1

            optimizer.zero_grad()
            net.zero_grad()
            output=net(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            loss = criterion(output, target)
            train_loss += loss.item()
            loss.backward()
            k=0
            same_num=0
            for i in net.modules():
                if isinstance(i, nn.Conv2d) or isinstance(i, nn.Linear):
                    m=torch.where(i.weight.grad*gr2[k]<0,1,0)
                    same_num+=torch.where(i.weight.grad*gr2[k]>0,1,0).sum().item()
                    i.weight.grad.add_(m*gr2[k]*beta)
                    k+=1

            optimizer.step()
            if (batch_idx % 100 == 0):
                _lr = lr_scheduler.get_last_lr()
                print('train[epoch:{}]-----loss:{:.8f},acc:{:.2f}%,lr:{}'.format(epoch, train_loss / total,correct / total * 100, _lr))
            mb.apply_masks()
        lr_scheduler.step()
        loss,acc=test(net, testloader, criterion, epoch)
        loss_test.append(loss)
        acc_test.append(acc)
    np.save('new_loss.npy', loss_test)
    np.save('new_test.npy', acc_test)



