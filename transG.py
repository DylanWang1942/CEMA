import os
import scipy.io as scio
from scipy.sparse import csr_matrix
def loadOurGraph(dataName):
    rows=[]
    cols=[]
    datas=[]
    shape=None
    with open('./dataset/ours/{}_network.txt'.format(dataName),'r') as f:
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
    ret_csr_matrix=csr_matrix((datas, (rows, cols)), shape=(shape, shape))
    return ret_csr_matrix
#gname = ["citeseer","20NG","blog","flickr","washington","blogcatalog","new_citeseer","wine","texas"]
gname = ["ppi"]
def loadFlickr():
    from scipy.sparse import load_npz
    return load_npz("./dataset/flickr/adj_full.npz")
for i in gname:
    csr_mat = loadOurGraph(i)
    scio.savemat("./dataset/ourOriginDataMatFormat/"+i+"_network.mat",{"graph_sparse":csr_mat})
    scio.savemat("./mat_data/"+i+"_network.mat",{"graph_sparse":csr_mat})

#dataFile = './dataset/ourOriginDataMatFormat/blogcatalog_network.mat'
#dataFile = './test.mat'
#data = scio.loadmat(dataFile)
#print(data)
