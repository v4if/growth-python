#!coding:utf-8
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

'''
k-mean聚类算法数据挖掘，将数据进行聚类分析，实验数据为data.txt
'''

marker_color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
marker_size = 121
maker_alpha = 0.4
# 聚类cluster K
K = 5


def assert_config():
    if len(marker_color) <= K:
        raise Exception("Marker color is insufficient")


def plt_help(allocate, data_set, plt_type='allocate'):
    if plt_type == 'allocate':
        # 原图像
        plt.subplot(1, 2, 1)
        plt.title("raw data")
        plt.scatter(data_set[:, 0], data_set[:, 1], color=marker_color[0], s=marker_size, alpha=maker_alpha)
        # 聚类之后的图像
        plt.subplot(1, 2, 2)
        for i in xrange(0, K):
            cluster_list = find(allocate[:], i)
            plt.scatter(data_set[cluster_list, 0], data_set[cluster_list, 1], color=marker_color[i], s=marker_size,
                        alpha=maker_alpha)
    elif plt_type == 'data_set':
        # 聚类之后的质心
        plt.subplot(1, 2, 2)
        for i in xrange(0, K):
            plt.scatter(data_set[i, 0], data_set[i, 1], color=marker_color[i], s=marker_size, marker='v',
                        alpha=maker_alpha)
    elif plt_type == 'title':
        plt.subplot(1, 2, allocate)
        plt.title(data_set)
    elif plt_type == 'show':
        plt.show()


# 欧式距离
def dist(data_set, centroid):
    return sum(pow((data_set - centroid), 2))


# 查找质心中所对应的样本
def find(cluster_allocate, i):
    tunnel_list = []
    iterator = 0
    for tunnel in cluster_allocate:
        if tunnel == i:
            tunnel_list.append(iterator)
        iterator += 1
    return tunnel_list


def k_mean(data_set, iterator_time):
    row, col = data_set.shape

    # 存储质心矩阵
    centroid = np.zeros((K, col))
    # 随机初始化质心矩阵种子
    for i in xrange(0, col):
        low = min(data_set[:, i])
        tunnel_range = max(data_set[:, i]) - low
        centroid[:, i] = np.repeat(low, K, axis=0) + tunnel_range * np.random.rand(K)

    # 用于存储每个样本被分配到的簇cluster
    allocate = np.zeros((row, 1))
    allocate.fill(-1)
    changed = True
    iterator = 0
    while changed:
        changed = False
        iterator += 1
        # 将样本分配给距离其最近的cluster
        for i in xrange(0, row):
            # 将距离第一个质心的距离作为当前最小值
            min_dist = dist(data_set[i, :], centroid[0, :])
            min_index = 0
            for j in xrange(1, K):
                cur_dist = dist(data_set[i, :], centroid[j, :])
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    min_index = j
            # 样本所在的cluster发生改变
            if min_index != allocate[i]:
                changed = True
                allocate[i] = min_index

        # 更新每个cluster的质心
        if changed:
            for i in xrange(0, K):
                cluster_list = find(allocate[:], i)
                if len(cluster_list) > 0:
                    centroid[i, :] = sum(data_set[cluster_list, :]) / len(cluster_list)

        # 0 不限制迭代次数
        if (iterator_time > 0) & (iterator >= iterator_time):
            break

    plt_sub_title = 'after {iter} iterator && K = {clu}'.format(iter=iterator, clu=K)
    plt_help(2, plt_sub_title, 'title')
    plt_help(allocate, data_set)
    return centroid


if __name__ == '__main__':
    # 颜色种类数要大于K的聚类数，还有一个颜色用于着色质心，便于着色区分
    assert_config()

    data_test = np.loadtxt("data.txt")

    cluster = k_mean(data_test, 0)
    plt_help(None, cluster, plt_type='data_set')
    plt_help(None, None, plt_type='show')
