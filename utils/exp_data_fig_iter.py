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
is_outlier = False
is_legend = False
# ========  iter vgg19/cifar10 =========
title = 'VGG19/CIFAR10'
data_baseline = 0.942
min_y, max_y = 0.905, 0.945
data_path = '../runs/exp_data/iter_vgg.xls'
xlim0 = 0.002
xlim1 = 0.1
ylim0 = 0.82
ylim1 = 0.945
# ======== iter resnet18/cifar100 =========
# title = 'ResNet18/CIFAR100'
# data_baseline = 0.7855
# min_y, max_y = 0.6, 0.8
# data_path = '../runs/exp_data/iter_resnet.xls'
# xlim0 = 0.001
# xlim1 = 0.1
# ylim0 = 0.48
# ylim1 = 0.8
# # ======== iter lenet5/mnist =========
# title = 'LeNet5/MNIST'
# data_baseline = 0.994
# min_y, max_y = 0.98, 0.995
# data_path = '../runs/exp_data/iter_lenet.xls'
# # 局部放大区域
# xlim0 = 0.001
# xlim1 = 0.1
# ylim0 = 0.96
# ylim1 = 0.995

exp_data = xlrd.open_workbook(data_path).sheets()[0]
# print(exp_data.ncols, exp_data.nrows)

# prune_ratio = np.array(exp_data.col_values(0, start_rowx=1, end_rowx=None))/100
prune_ratio = 1-np.array(exp_data.col_values(0, start_rowx=1, end_rowx=None))/100
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

    def __init__(self, xpixels=500, ypixels=500, dpi=None, nrows=1, ncols=1):
        import matplotlib.pyplot as plt
        plt.rcParams['axes.unicode_minus'] = False  # -
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rc('font', family='Times New Roman')
        if dpi is None:
            fig, ax = plt.subplots(nrows, ncols)
        else:
            xinch = xpixels / dpi
            yinch = ypixels / dpi
            fig, ax = plt.subplots(nrows, ncols, figsize=(xinch, yinch))
        self.plt = plt
        self.ax = ax
        # self.color = ['red', 'orange', 'yellow', 'green', 'blue', 'cyan', 'purple', 'pink', 'coral', 'gold', 'lime', 'navy', 'teal', 'indigo']
        self.color = ['red', 'green', 'blue', 'cyan', 'purple', 'pink', 'coral', 'gold', 'lime', 'navy', 'teal', 'indigo']


linewidth = 2
fig = PltClass(xpixels=600, ypixels=300, dpi=60)
legend = []
fig.ax.axhline(data_baseline, color='gray', linestyle='-', linewidth=3, alpha=0.8)
for i in range(len(label_class)):
    legend.append(label_class[i])
    fig.ax.plot(prune_ratio, accuracy[i], color=fig.color[i], linestyle='-', linewidth=linewidth, alpha=0.8)
    fig.ax.fill_between(prune_ratio, accuracy[i]-accuracy_std[i], accuracy[i]+accuracy_std[i], color=fig.color[i], alpha=0.2)
fig.plt.xscale('log')

# 自定义legend
fontsize = 20
if is_legend:
    import matplotlib.patches as mpatches
    rank_algo = [mpatches.Patch(color=fig.color[i], label="{:s}".format(legend[i])) for i in range(len(legend))]
    patches = rank_algo
    fig.ax.legend(handles=patches, fontsize=fontsize)

fig.plt.title(title, fontsize=fontsize)
fig.plt.xlabel('Remaining ratio', fontsize=fontsize)
fig.plt.ylabel('Test accuracy', fontsize=fontsize)

# plt.legend(loc='upper left')
fig.plt.tick_params(labelsize=fontsize)
fig.plt.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.15)
fig.plt.grid(linestyle=':')  # 生成网格
if is_outlier:
    fig.plt.ylim([min_y, max_y])


# === 局部放大显示 ===
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# 嵌入绘制局部放大图的坐标系
axins = inset_axes(fig.ax, width="40%", height="60%", loc='lower left',
                   bbox_to_anchor=(0.5, 0.1, 1, 1),
                   bbox_transform=fig.ax.transAxes)

# 在子坐标系中绘制原始数据
axins.axhline(data_baseline, color='gray', linestyle='-', linewidth=3, alpha=0.8)
for i in range(len(label_class)):
    axins.plot(prune_ratio, accuracy[i], color=fig.color[i], linestyle='-', linewidth=linewidth, alpha=0.8)
    axins.fill_between(prune_ratio, accuracy[i]-accuracy_std[i], accuracy[i]+accuracy_std[i], color=fig.color[i], alpha=0.2)
axins.set_xscale('log')
axins.tick_params(labelsize=fontsize*0.8)

# 调整子坐标系的显示范围
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)

# 建立父坐标系与子坐标系的连接线
# loc1 loc2: 坐标系的四个角
# 1 (右上) 2 (左上) 3(左下) 4(右下)
mark_inset(fig.ax, axins, loc1=2, loc2=1, fc="none", ec='k', lw=1)

fig.plt.show()
