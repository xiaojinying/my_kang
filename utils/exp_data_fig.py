# -*- coding: utf-8 -*-

"""
Created on 04/26/2022
exp_data_fig.
@author: Kang Xiatao (kangxiatao@gmail.com)
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xlrd, xlwt


smooth = False
is_outlier = True
is_legend = True
# ========  vgg19/cifar10 =========
# title = 'VGG19/CIFAR10'
# data_baseline = 0.942
# min_y, max_y = 0.905, 0.945
# data_path = '../runs/exp_data/cifar10_vgg19.xls'
# ========  resnet18/cifar100 =========
# title = 'ResNet18/CIFAR100'
# data_baseline = 0.7855
# min_y, max_y = 0.65, 0.8
# data_path = '../runs/exp_data/cifar100_resnet18.xls'
# ========  lenet5/mnist =========
title = 'LeNet5/MNIST'
data_baseline = 0.994
min_y, max_y = 0.98, 0.995
data_path = '../runs/exp_data/mnist_lenet5.xls'

exp_data = xlrd.open_workbook(data_path).sheets()[0]
# print(exp_data.ncols, exp_data.nrows)

prune_ratio = np.array(exp_data.col_values(0, start_rowx=1, end_rowx=None))/100
# prune_ratio = 1-np.array(exp_data.col_values(0, start_rowx=1, end_rowx=None))/100
label = exp_data.row_values(0, start_colx=1, end_colx=None)
label_class = list(set(label))
label_class.sort()
# print(label_class)
accuracy = []
accuracy_std = []


# 差值平滑
def inter_smooth(x, y, t=300):
    from scipy.interpolate import make_interp_spline
    x_smooth = np.linspace(x.min(), x.max(), t)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return x_smooth, y_smooth


# 取出每个类对应的精度，并求均值
for lab in label_class:
    acc = []
    for i, l in enumerate(label):
        if l == lab:
            if smooth:
                temp = exp_data.col_values(i+1, start_rowx=1, end_rowx=None)
                x, y = inter_smooth(prune_ratio, np.array(temp), 10)
                acc.append(y)
            else:
                acc.append(exp_data.col_values(i+1, start_rowx=1, end_rowx=None))
    # print(acc)
    accuracy.append(np.mean(np.array(acc), axis=0)/100)
    accuracy_std.append(np.std(np.array(acc)/100, axis=0))
if smooth:
    prune_ratio = x
# print(accuracy)


# 作图
class PltClass(object):

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
        self.color = ['red', 'green', 'blue', 'cyan', 'purple', 'pink', 'coral', 'gold', 'lime', 'navy', 'teal', 'indigo']


linewidth = 2
fig = PltClass(xpixels=600, ypixels=300, dpi=60)
legend = []
fig.plt.axhline(data_baseline, color='gray', linestyle='-', linewidth=3, alpha=0.8)
for i in range(int(len(label_class)/2)):
    legend.append(label_class[i*2])
    fig.plt.plot(prune_ratio, accuracy[i*2], color=fig.color[i], linestyle='-', linewidth=linewidth, alpha=0.8)
    fig.plt.fill_between(prune_ratio, accuracy[i*2]-accuracy_std[i*2], accuracy[i*2]+accuracy_std[i*2], color=fig.color[i], alpha=0.2)
    fig.plt.plot(prune_ratio, accuracy[i*2+1], color=fig.color[i], linestyle='-.', linewidth=linewidth, alpha=0.8)
    fig.plt.fill_between(prune_ratio, accuracy[i*2+1]-accuracy_std[i*2+1], accuracy[i*2+1]+accuracy_std[i*2+1], color=fig.color[i], alpha=0.2)


# fig.plt.xscale('log')
# plt.gca().set_xticks(data_x)
# plt.gca().set_xticklabels(scale_x)


# 自定义legend
fontsize = 20
if is_legend:
    import matplotlib.patches as mpatches
    mode_label = ['Rank', 'Random']
    mode_linestyle = ['-', '-.']
    rank_algo = [mpatches.Patch(color=fig.color[i], label="{:s}".format(legend[i])) for i in range(len(legend))]
    prune_mode = [fig.plt.plot([], [], color='k', linestyle=mode_linestyle[i], linewidth=linewidth, label=mode_label[i])[0] for i in range(len(mode_label))]
    patches = rank_algo + prune_mode
    fig.plt.legend(handles=patches, fontsize=fontsize)

fig.plt.title(title, fontsize=fontsize)
fig.plt.xlabel('Pruning ratio', fontsize=fontsize)
fig.plt.ylabel('Test accuracy', fontsize=fontsize)

# plt.legend(loc='upper left')
fig.plt.tick_params(labelsize=fontsize)
fig.plt.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.15)
fig.plt.grid(linestyle=':')  # 生成网格
if is_outlier:
    fig.plt.ylim([min_y, max_y])
fig.plt.show()
