import numpy as np
import torch.utils.data as Data
import torch
import pickle
import scipy.sparse as sp


# 加载我们的图
# 将邻接矩阵格式转化为我们的格式
def loadOurGraph(datasets):
    rows=[]
    cols=[]
    datas=[]
    shape=None
    with open('../dataset-adjacent/our_reduce_graph/{}.txt'.format(datasets),'r') as f:
        lines=f.readlines()
        for line in lines:
            line=line[:-1]
            datalist=line.split(' ')
            datalist[0]=int(eval(datalist[0]))
            datalist[1]=int(eval(datalist[1]))
            datalist[2]=eval(datalist[2])
            rows.append(datalist[0])
            cols.append(datalist[1])
            datas.append(datalist[2])
        shape=max(rows+cols)+1
        print([shape,shape])
        adjmatrix=np.zeros([shape,shape])
        for i in range(len(lines)):
            adjmatrix[rows[i]][cols[i]]=datas[i]
        return adjmatrix,[shape,shape]

def transInto(adjMatrix):
    shape=len(adjMatrix)
    labels=torch.from_numpy(np.array([-1]*shape))
    # print(type(labels))
    print(adjMatrix)
    adjMatrix=adjMatrix.astype(np.int32)
    adjMatrix=torch.from_numpy(adjMatrix)

    # print(type(adjMatrix))
    torch_dataset = Data.TensorDataset(adjMatrix,labels)
    # print(torch_dataset)
    return torch_dataset


def cootransInto(adj):
    shape=adj.shape[0]
    labels=torch.from_numpy(np.array([-1]*shape))
    # print(type(labels))
    adj=adj.astype(np.int32)
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = torch.sparse_coo_tensor([adj.row,adj.col], adj.data,(shape,shape))
    print(adj)
    #adj=torch.from_numpy(adj)
    # print(type(adj))
    torch_dataset = Data.TensorDataset(adj,labels)
    # print(torch_dataset)
    return torch_dataset














