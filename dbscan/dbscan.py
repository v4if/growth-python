#!coding:utf-8
from __future__ import division
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

'''
sudo apt-get install python-matplotlib
sudo apt-get install python-scipy
'''

marker_color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
marker_size = 121
maker_alpha = 0.4


# 查找每个簇所对应的样本
def find(data, i):
    tunnel_list = []
    iterator = 0
    for tunnel in data[:, -2]:
        if tunnel == i:
            tunnel_list.append(iterator)
        iterator += 1
    return tunnel_list


def plt_help(data, cluster):
    plt.subplot(1, 2, 1)
    plt.title("raw data")
    plt.scatter(data[:, 0], data[:, 1], color=marker_color[0], s=marker_size, alpha=maker_alpha)

    plt.subplot(1, 2, 2)
    plt.title("rend data")
    for i in xrange(0, cluster + 1):
        tunnel_list = find(data, i)
        plt.scatter(data[tunnel_list, 0], data[tunnel_list, 1], color=marker_color[i % len(marker_color)],
                    s=marker_size, alpha=maker_alpha)
    tunnel_list = find(data, -1)
    plt.scatter(data[tunnel_list, 0], data[tunnel_list, 1], color=marker_color[-1], s=marker_size, marker='v',
                alpha=maker_alpha)
    plt.show()


def dbscan(data_set, eps, min_pts):
    row, col = data_set.shape

    # 聚类的簇的个数
    cluster = -1
    # 样本分配到哪个簇
    allocate = np.zeros((row, 1))
    allocate.fill(-1)
    # 是否已经访问过
    visited = np.zeros((row, 1))
    visited.fill(False)

    # [raw_data, allocate, visited]
    data_set = np.hstack((data_set, allocate, visited))
    for p in data_set:
        if not p[-1]:
            p[-1] = True
            neighbor_pts = region_query(p, data_set, eps)

            # 标记为噪声点
            if len(neighbor_pts) < min_pts:
                p[-2] = -1
            else:
                cluster += 1
                # 从某个核心点出发，不断向密度可达的区域扩张，从而得到一个包含核心点和边界点的最大化区域，区域中任意两点密度相连
                p[-2] = cluster
                for p_neighbor in neighbor_pts:
                    if not p_neighbor[-1]:
                        p_neighbor[-1] = True
                        p_neighbor_pts = region_query(p_neighbor, data_set, eps)
                        if len(p_neighbor_pts) >= min_pts:
                            # 加入邻域点
                            for p_add in p_neighbor_pts:
                                neighbor_pts.append(p_add)
                    if p_neighbor[-2] == -1:
                        p_neighbor[-2] = cluster
    return data_set, cluster


# 欧式距离
def dist(f_p, t_p):
    return sum(pow((f_p[0:2] - t_p[0:2]), 2))


# 计算核心点邻域
def region_query(p, d, eps):
    neighbor_pts = []
    for t_p in d:
        if dist(p, t_p) <= eps:
            neighbor_pts.append(t_p)
    return neighbor_pts


if __name__ == '__main__':
    flag = 'cluster'
    if flag == 'test':
        np_data = np.array([[2, 5.4], [2, 6], [3, 5.7], [6, 2], [7, 1.8], [4, 5.6]])
        arg_eps = 2
        arg_minPts = 2
        rend_data, c = dbscan(np_data, arg_eps, arg_minPts)
        plt_help(rend_data, c)
    elif flag == 'cluster':
        mat_contents = sio.loadmat('data/moon.mat')
        np_data = np.array(mat_contents['a'])

        arg_eps = 0.02
        arg_minPts = 3
        rend_data, c = dbscan(np_data, arg_eps, arg_minPts)
        plt_help(rend_data, c)
    elif flag == 'dist':
        print dist(np.array([0.294086, 12.097]), np.array([0.0456, 11.9015]))
        print dist(np.array([-0.4924, 12.06]), np.array([-0.3978, 12.1051]))
        print dist(np.array([0.589, 12.034]), np.array([0.708, 12.048]))
        print dist(np.array([2, 5.4]), np.array([4, 5.6]))
        print dist(np.array([3, 5.7]), np.array([4, 5.6]))
