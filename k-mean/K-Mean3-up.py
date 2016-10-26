#!coding:utf-8
from __future__ import division
import numpy as np
import random

'''
k-mean聚类算法图像分割，将图像分割并着色，实验数据为favorite.png
整合up版
'''

# 聚类cluster K
K = 10
use_coord_stigma = True
# 坐标特征值相对于像素值的权重，即coord * coord_K
coord_K = 0


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

    print 'cluster iterator : {c_it}'.format(c_it=iterator)
    return allocate


def im_to_np(im):
    row, col, pixel = im.shape
    if use_coord_stigma:
        # 添加坐标作为样本特征进行聚类分析
        np_row_index = np.repeat(0, col, axis=0) * coord_K
        np_col_index = np.arange(col) * coord_K
        np_coord_index = np.hstack((np_row_index[:, np.newaxis], np_col_index[:, np.newaxis]))
        im_np = np.hstack((im[0, :], np_coord_index))
        for i in xrange(1, row):
            np_row_index = np.repeat(i, col, axis=0) * coord_K
            np_col_index = np.arange(col) * coord_K
            np_coord_index = np.hstack((np_row_index[:, np.newaxis], np_col_index[:, np.newaxis]))
            t = np.hstack((im[i, :], np_coord_index))
            im_np = np.vstack((im_np, t))
    else:
        im_np = np.array(im[0, :])
        for i in xrange(1, row):
            t = np.array(im[i, :])
            im_np = np.vstack((im_np, t))

    return im_np


def im_rending(r_np_data, allocate):
    # 产生随机着色颜色
    color = []
    for i in xrange(0, K):
        color.append((random.randrange(256), random.randrange(256), random.randrange(256)))

    clu_k = 0
    for i in xrange(0, K):
        cluster_list = find(allocate[:], i)
        if len(cluster_list) > 0:
            clu_k += 1
            if use_coord_stigma:
                r_np_data[cluster_list, :-2] = color[i]
            else:
                r_np_data[cluster_list, :] = color[i]
    print 'cluster k : {c_k}'.format(c_k=clu_k)
    print r_np_data
    return r_np_data


def np_to_im(np_data, im_data):
    row, col, pixel = im_data.shape
    split_array = np.split(np_data, row, axis=0)
    for i in xrange(0, row):
        if use_coord_stigma:
            # [[2 3 4 0 0], [4 5 6 0 1]] 第一维全取，第二维取到倒数第二个元素
            # 舍弃掉坐标特征，转换为pixel像素值
            im_data[i, :] = split_array[i][:, :-2]
        else:
            im_data[i, :] = split_array[i]
    return im_data

if __name__ == '__main__':
    import cv2

    debug_tag = False

    if debug_tag:
        raw_im_data = np.array([[[2, 3, 4], [4, 5, 6]], [[7, 8, 9], [1, 2, 3]]])
        add_item = np.array([[0, 0], [0, 1]])

        row_index = np.repeat(1, raw_im_data.shape[1], axis=0)
        col_index = np.arange(raw_im_data.shape[1])
        coord_index = np.hstack((row_index[:, np.newaxis], col_index[:, np.newaxis]))
        main_np_data = im_to_np(raw_im_data)
        print raw_im_data
        print main_np_data
        main_alloc = np.array([1, 0, 0, 2])
        im_rending(main_np_data, main_alloc)
        row_data = main_np_data[0]
        print np_to_im(main_np_data, raw_im_data)
    else:
        raw_im_data = cv2.imread("favorite.png")
        raw_np_data = im_to_np(raw_im_data)
        alloc = k_mean(raw_np_data, 2)
        rend_np_data = im_rending(raw_np_data, alloc)
        rend_im_data = np_to_im(rend_np_data, raw_im_data)
        cv2.imshow("Rend", rend_im_data)  # 显示图像
        cv2.waitKey(0)  # 程序暂停
        cv2.waitKey(0)  # 程序暂停
