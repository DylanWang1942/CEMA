import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
import scipy.io as sio
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch
import torch.nn.functional as F
import pandas as pd
# from sklearn.metrics import normalized4_mutual_info_score as nmi
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score as acc
from itertools import cycle
import matplotlib.colors as colors
from utils import *
import sklearn.metrics as sm
from graph import Graph

import matplotlib as mpl
import os

mpl.use('Agg')  # mpl.use('TkAgg')
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize

from gensim.models import Word2Vec, KeyedVectors


# 获得相似度（向量内积）
def getSimilarity(result):
    return np.dot(result, result.T)


# 网络重构的precision@k
def check_reconstruction(embedding, graph_data, check_index):
    def get_precisionK(embedding, data, max_index):
        print("get precisionK...")
        similarity = getSimilarity(embedding).reshape(-1)
        sortedInd = np.argsort(similarity)
        cur = 0
        count = 0
        precisionK = []
        sortedInd = sortedInd[::-1]
        for ind in sortedInd:
            x = ind / data.N
            y = ind % data.N
            count += 1
            if (data.adj_matrix[x].toarray()[0][y] == 1 or x == y):
                cur += 1
            precisionK.append(1.0 * cur / count)
            if count > max_index:
                break
        return precisionK

    precisionK = get_precisionK(embedding, graph_data, np.max(check_index))
    ret = []
    for index in check_index:
        print("precisonK[%d] %.2f" % (index, precisionK[index - 1]))
        ret.append(precisionK[index - 1])
    return ret


# AUC
def check_link_prediction(X, Y, test_ratio=0.1):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_ratio)

    for im in range(y_test.shape[1]):

        if len(np.unique(y_test[:, im])) != 2:
            print('未采样到所有的类，请重新运行算法')
            print(len(np.unique(y_test[:, im])))
            exit(0)

    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(x_train, y_train)
    y_pred = clf.predict_proba(x_test)

    auc = roc_auc_score(y_test, y_pred)
    '''
    y_train = np.argwhere(y_train==1)[:,1]
    y_test = np.argwhere(y_test==1)[:,1]

    lg = LogisticRegression(penalty='l2',C=0.001)
    lg.fit(x_train,y_train)
    lg_y_pred_est = lg.predict_proba(x_test)[:,1]
    fpr,tpr,thresholds = sm.roc_curve(y_test,lg_y_pred_est)
    #average_precision = average_precision_score(y_test, lg_y_pred_est)
    '''

    return auc


# 读取对应数据集的标签
def read_true_label(dataset_name):
    label = sio.loadmat('./label/' + dataset_name + '_label.mat')
    label = label['label']
    # print('The true labels size is {}'.format(np.array(label).size))
    return label


# nmi值，数据可视化，输入：
# 表示向量
# 真实标签
# 聚类簇数
def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


# 输出nmi acc
def cluster(data, true_labels, n_clusters=3):
    # 对数据各种预处理，看情况添加
    # data = scale(data)
    # min_max_scaler = MinMaxScaler()
    # data = min_max_scaler.fit_transform(data)
    # data = normalize(data, norm = 'l2')

    km = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    km.fit(data)
    km_means_labels = km.labels_  # 这里获得的label数小于真实label
    km_means_cluster_centers = km.cluster_centers_
    km_means_labels_unique = np.unique(km_means_labels)

    true_labels = true_labels - 1
    true_labels = np.squeeze(true_labels)
    true_labels = [i - 1 for i in true_labels]
    # print("true label size is")
    # print(true_labels)
    # print(np.array(true_labels).size)
    # print("km_means_labels size is")
    # print(km_means_labels.size)
    nmi1 = nmi(true_labels, km_means_labels)  # 这里的标签数量不对报错了
    acc = cluster_acc(np.array(true_labels), km_means_labels)

    # print('nmi', nmi1)
    # print('acc', acc)
    return nmi1, acc


def check_clusters(reprsn, Y, test_ratio=0.9, dataName='Cora'):
    dataRes = {dataName: [[], []]}
    # 测试聚类

    nmi, acc = cluster(reprsn, Y, Q_net.n_classes)
    dataRes[dataName][0].append(nmi)
    dataRes[dataName][1].append(acc)
    pass


def exam_clusters(dataset_name, reprsn, index, log_name, file_name):
    train_graph_data = Graph('./mat_data/' + dataset_name + '_network.mat', 0.0)
    # print("N:",train_graph_data.N)
    train_graph_data.load_label_data(
        './label/' + dataset_name + '_label.txt')
    data = train_graph_data.sample(
        train_graph_data.N, do_shuffle=False, with_label=True)
    # print("label shape:",data.label.shape)
    num = 0.9
    check_clusters(reprsn, data.label, test_ratio=num)

    pass


# 多标签分类 f 值
# test_ratio = 0.9 表示使用90%的数据进行测试，使用10%的数据进行训练
def check_multi_label_classification(X, Y, test_ratio=0.9):
    def small_trick(y_test, y_pred):
        y_pred_new = np.zeros(y_pred.shape, np.bool)
        sort_index = np.flip(np.argsort(y_pred, axis=1), 1)
        for i in range(y_test.shape[0]):
            num = sum(y_test[i])
            # print(num)
            for j in range(num):
                y_pred_new[i][sort_index[i][j]] = True
        return y_pred_new

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_ratio)
    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(x_train, y_train)
    y_pred = clf.predict_proba(x_test)
    # print(y_pred)
    y_pred = small_trick(y_test, y_pred)
    # print(y_pred)
    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")


    log  = OneVsRestClassifier(LogisticRegression())
    log.fit(x_train,y_train)
    pred_labels = (log.predict(x_test)).tolist()
    acc = accuracy_score(y_test, np.around(pred_labels))
    return micro, macro, acc




# 本实验网络结构
# 用于读取模型
# 需要根据数据维数调整参数
##################################
# Define Networks
##################################
# Encoder
data_dict = {
    'cora': {
        'n_classes': 7,
        'z_dim': 128,
        'X_dim': 2708,
        'N': 1024
    },
    'citeseer': {
        'n_classes': 7,
        'z_dim': 128,
        'X_dim': 3327,
        'N': 1024
    },
    'pubmed': {
        'n_classes': 3,
        'z_dim': 128,
        'X_dim': 19717,
        'N': 2048
    }

}


class Q_net(nn.Module):
    n_classes = 0
    z_dim = 0
    X_dim = 0
    N = 0

    def __init__(self):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(Q_net.X_dim, Q_net.N)
        self.lin2 = nn.Linear(Q_net.N, Q_net.N)
        # Gaussian code (z)
        self.lin3gauss = nn.Linear(Q_net.N, Q_net.z_dim)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        xgauss = self.lin3gauss(x)

        return xgauss


# 测试函数
# 输入：
# 数据集名称
# 表示向量
# 原始网络
# patk参数
# 输出文件夹的名字

def exam_outputs(dataset_name, reprsn, index, log_name,file_name):
    train_graph_data = Graph('./mat_data/' + dataset_name + '_network.mat', 0.0)
    # print("N:",train_graph_data.N)
    train_graph_data.load_label_data(
        './label/' + dataset_name + '_label.txt')

    # p_at_k 适用于部分算法，暂时不用
    # p_at_k = check_reconstruction(reprsn, train_graph_data, index)
    p_at_k = 0

    data = train_graph_data.sample(
        train_graph_data.N, do_shuffle=False, with_label=True)
    # print("label shape:",data.label.shape)

    micro_macro = ''
    acc_string = ''
    AUC = ''
    f_test_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    max_f_test_ratio = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91]

    for num in f_test_ratio:
        micro = []
        macro = []
        acc = []
        auc = []
        micro_macro += str(num) + ': \n'
        acc_string += str(num) + ': \n'
        for k in range(10):
            temp_micro, temp_macro, temp_acc = check_multi_label_classification(reprsn, data.label, test_ratio=num)
            micro.append(temp_micro)
            macro.append(temp_macro)
            acc.append(temp_acc)
        micro_mean = np.mean(micro)
        macro_mean = np.mean(macro)
        acc_mean = np.mean(acc)
        micro_std = np.std(micro, ddof=1)
        macro_std = np.std(macro, ddof=1)
        acc_std = np.std(acc, ddof=1)
        micro_macro += str(micro_mean) + ' ' + str(micro_std) + '\n' + str(macro_mean) + ' ' + str(macro_std) + '\n'
        acc_string += str(acc_mean) + ' ' + str(acc_std) + '\n'
        AUC += str(num) + ': \n'
        for k in range(10):
            temp2 = check_link_prediction(reprsn, data.label, test_ratio=num)
            auc.append(temp2)
        auc_mean = np.mean(auc)
        auc_std = np.std(auc)
        AUC += str(auc_mean) + ' ' + str(auc_std) + '\n'

    # 1-5%
    """
    for num in max_f_test_ratio:
        temp = check_multi_label_classification(reprsn, data.label, test_ratio=num)
        micro_macro += str(num) + ': ' + str(temp) + '\n'
        temp2 = check_link_prediction(reprsn,data.label,test_ratio=num)
        AUC += str(num) +': ' + str(temp2) + '\n'
    """
    print("saving the outcome in " + '../logs/' + log_name + '/' + file_name + '.txt')
    if (os.path.exists('./logs/' + log_name) == False):
        os.mkdir('./logs/' + log_name)
    if (os.path.exists('./logs/' + log_name ) == False):
        os.mkdir('./logs/' + log_name )
    with open('./logs/' + log_name  + '/' + file_name + '.txt', 'w+') as f:
        f.write(log_name + '\n' + dataset_name + '\n' + 'p@k: ' + str(index) +
                '  ' + str(p_at_k) + '\n' + 'micro_macro:\n ' + str(micro_macro) + '\n' + 'acc:\n ' + str(acc_string) \
                + '\n' + 'AUC: \n' + str(AUC))


def evaluation(reprsn, dataName, level, propa, fusion, community):
    if (propa):
        propa = "propa"
    else:
        propa = "nopropa"
    if (fusion):
        fusion = "fusion"
    else:
        fusion = "nofusion"
    if (community):
        community = "community"
    else:
        community = "basic"
    '''
    :param reprsn: 嵌入式表示
    :param dataName: 数据集名称
    '''
    # 检测本文提出的算法
    indexLabel = [2, 10, 100, 200, 300, 500, 800, 1000]

    dataRes = {dataName: [[], []]}
    save_file_name = dataName
    try:
        data_dict[dataName]
    except:
        print('找不到该数据集')
        exit(0)

    Q_net.n_classes = data_dict[dataName]['n_classes']
        # Q_net.z_dim = data_dict[dataName]['z_dim']
        # Q_net.X_dim = data_dict[dataName]['X_dim']
        # Q_net.N = data_dict[dataName]['N']
        # 测试点分类、连接预测
    exam_outputs(dataName, reprsn, indexLabel, 
                     save_file_name + "_" + propa + "_" + community + "_" + fusion + "_" + str(
                         level))  # exam_outputs(dataName, reprsn, indexLabel, 'cadne', save_file_name)

    for time in range(10):
        # 测试聚类
        label = read_true_label(dataName)
        nmi, acc = cluster(reprsn, label, Q_net.n_classes)
        dataRes[dataName][0].append(nmi)
        dataRes[dataName][1].append(acc)
    print('-------------')

    for key in dataRes.keys():
        nmiMean = np.array(dataRes[key][0]).mean()
        nmiStd = np.array(dataRes[key][0]).std(ddof=1)

        accMean = np.array(dataRes[key][1]).mean()
        accStd = np.array(dataRes[key][1]).std(ddof=1)
        # print(key)
        # print('acc',accMean)
        # print('nmi',nmiMean)
        # print('-------------')
    logaccnmi(
              save_file_name + "_" + propa + "_" + community + "_" + fusion + "_" + str(
                  level),  dataName, nmiMean, accMean, nmiStd, accStd)
    return accMean, nmiMean


def logaccnmi(log_name, file_name, dataset_name, nmi, acc, nmiStd, accStd):
    with open('./logs/' + log_name + '/' + file_name + '.txt', 'a') as f:
        f.write('\n' + 'acc: ' + str(acc) + ' ' + str(accStd) + '\n' + 'nmi: ' + str(nmi) + ' ' + str(nmiStd) + '\n')
