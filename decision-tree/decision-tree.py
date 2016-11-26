#! /usr/bin/python2.7
# coding:utf-8

import numpy as np
import math

'''
熵的解释：体系混乱程度的体现，至少需要多少位编码才能表示该信息
属性、类别标签，根据信息熵的增益度构造决策树
'''


# 计算数据的熵值
def entropy(data_set):
    # 字典，用于存储类别标签
    label = {}
    for d in data_set:
        l = d[-1]
        if l not in label.keys():
            label[l] = 0
        label[l] += 1

    t = len(data_set)
    ret = 0
    for a in label:
        pi = float(label[a]) / t
        ret -= pi * math.log(pi, 2)
    return ret


# 决策树叶子节点含有正反例，投票决定当前标签的值
def vote_label_value(data_set):
    # 字典，用于存储类别标签
    label = {}
    for d in data_set:
        l = d[-1]
        if l not in label.keys():
            label[l] = 0
        label[l] += 1
    result = sorted(label.items())
    return result[-1][0]


# 以当前标签某个属性分类的子树，并剔除该index列，以实现决策树递归
def find_to_split(data_set, key, index):
    result = []
    if data_set[0, index].isdigit():
        v1 = [float(v0) for v0 in data_set[:, index]]
        step = np.mean(v1)
        for i in data_set:
            if key == 'gt' and float(i[index]) >= step:
                v0 = [v1 for v1 in i[0: index]]
                v3 = [v2 for v2 in i[index + 1:]]
                v0.extend(v3)
                result.append(v0)
            elif key == 'lt' and float(i[index]) < step:
                v0 = [v1 for v1 in i[0: index]]
                v3 = [v2 for v2 in i[index + 1:]]
                v0.extend(v3)
                result.append(v0)
    else:
        for i in data_set:
            if i[index] == key:
                v0 = [v1 for v1 in i[0: index]]
                v3 = [v2 for v2 in i[index + 1:]]
                v0.extend(v3)
                result.append(v0)
    return result


# 计算最大增益属性的下标
def best_gain_index(data_set):
    whole_entropy = entropy(data_set)
    data_scale = len(data_set)
    max_gain = 0.0
    best_index = -1
    # 对每一个属性，对其进行划分，并计算信息增益
    for i in xrange(len(data_set[0]) - 1):
        cluster = []
        if data_set[0, i].isdigit():
            v1 = [float(v0) for v0 in data_set[:, i]]
            step = np.mean(v1)
            for j in xrange(data_scale):
                c = float(data_set[j, i])
                if c >= step and 'gt' not in cluster:
                    cluster.append('gt')
                elif 'lt' not in cluster:
                    cluster.append('lt')
        else:
            for j in xrange(data_scale):
                c = data_set[j, i]
                if c not in cluster:
                    cluster.append(c)

        # 计算总增益
        iterator_entropy = 0
        for k in cluster:
            v2 = find_to_split(data_set, k, i)
            sub_scale = len(v2)
            iterator_entropy += (float(sub_scale) / data_scale) * entropy(v2)
        # 判断当前划分增益是否大于max_gain
        if (whole_entropy - iterator_entropy) > max_gain:
            max_gain = whole_entropy - iterator_entropy
            best_index = i
    return best_index


# 递归创建决策树
def train_tree(data_set, attr):
    # 所有属性都已用来划分
    if len(attr) == 0:
        return vote_label_value(data_set)
    # 待划分空间内的所有样本属于同一类别
    if len(set(data_set[:, -1])) == 1:
        return data_set[0][-1]

    best_index = best_gain_index(data_set)
    if best_index == -1:
        return vote_label_value(data_set)
    best_label = attr[best_index]
    del(attr[best_index])

    best_tree = {best_label: {}}
    # 对每一属性值递归求决策树
    if data_set[0, best_index].isdigit():
        decision_labels = []
        v1 = [float(v0) for v0 in data_set[:, best_index]]
        step = np.mean(v1)
        for i in data_set:
            if float(i[best_index]) >= step and 'gt' not in decision_labels:
                decision_labels.append('gt')
            elif 'lt' not in decision_labels:
                decision_labels.append('lt')
        # 为当前子树添加分界的平均值
        best_tree['mean'] = step
    else:
        decision_labels = set(data_set[:, best_index])

    for v0 in decision_labels:
        sub_data_set = find_to_split(data_set, v0, best_index)
        sub_attr = attr[:]
        best_tree[best_label][v0] = train_tree(np.array(sub_data_set), sub_attr)

    return best_tree


# 遍历决策树
def traveled_tree(d_tree, data_set, attr):
    if type(d_tree).__name__ == 'string_':
        return d_tree

    root_key = (d_tree.keys())[-1]
    attr_index = attr.index(root_key)

    data = data_set[attr_index]
    if data.isdigit():
        step = d_tree['mean']
        if float(data) >= step:
            result = traveled_tree(d_tree[root_key]['gt'], data_set, attr)
        else:
            result = traveled_tree(d_tree[root_key]['lt'], data_set, attr)
    else:
        result = traveled_tree(d_tree[root_key][data], data_set, attr)

    return result


# 预测未来
def wake_feature(d_tree, data_set, attr):
    return traveled_tree(d_tree, data_set, attr)


if __name__ == '__main__':
    raw_data = np.loadtxt("data/test.txt", dtype={'names': ['Outlook', 'Temperature', 'Humidity', 'Windy', 'Play'],
                                                  'formats': ['S8', 'i4', 'i4', 'S5', 'S3']})
    data_attr = ['Outlook', 'Temperature', 'Humidity', 'Windy']

    list_data = [list(e) for e in raw_data]
    np_data = np.array(list_data)

    decision_tree = train_tree(np_data, data_attr)
    print decision_tree

    # test_data = np.array(['sunny', '85', '85', 'false'])
    # test_data = np.array(['sunny', '80', '90', 'true'])
    test_data = np.array(['sunny', '69', '70', 'false'])
    attr_label = ['Outlook', 'Temperature', 'Humidity', 'Windy']
    decision_feature = wake_feature(decision_tree, test_data, attr_label)
    print decision_feature
