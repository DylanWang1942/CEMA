import numpy as np
import networkx as nx
import os
import subprocess
from scipy.sparse import identity
from scipy.sparse import csr_matrix
from scipy.io import mmwrite,mmread
import sys
from argparse import ArgumentParser  
from sklearn.preprocessing import normalize 
import time
import warnings

import utils

warnings.filterwarnings("ignore")
                                                            
from utils import *
from scoring import lr
sys.path.append("./metric")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3s'
class IdRemapper(object):
    """Maps string Ids into 32 bits signed integer Ids starting from 0."""
    INT_MAX = 2147483647

    def __init__(self):
        """Construct a new Id Remapper class instance."""
        self.curr_id = 0
        self.mapping = {}
        self.r_mapping = {}

    def get_int_id(self, str_id):
        """Get a unique 32 bits signed integer for the given string Id."""
        if not str_id in self.mapping:
            if self.curr_id == IdRemapper.INT_MAX:
                return None # No more 32 bits signed integers available
            self.mapping[str_id] = self.curr_id
            self.r_mapping[self.curr_id] = str_id
            self.curr_id += 1
        return self.mapping[str_id]

    def get_str_id(self, int_id):
        """Get the original string Id for the given signed integer Id."""
        return self.r_mapping[int_id] if int_id in self.r_mapping else None
id_remapper = IdRemapper()

def graph_fusion(laplacian, feature, num_neighs, mcr_dir, coarse, fusion_input_path, \
                 search_ratio, fusion_output_dir, mapping_path, dataset):

    mapping = sim_coarse_fusion(laplacian)
    feats_laplacian = feats2graph(feature, num_neighs, mapping)
    fused_laplacian = laplacian + feats_laplacian
    return fused_laplacian

def refinement_concate(G,levels, projections, coarse_laplacian, embeddings_all, lda, power,q,propa,propacomm,communities):
    embeddings = embeddings_all[0].copy()#embeddings_all[levels]
    for i in range(levels):
        num_project = levels - i - 1
        embeddings = projections[num_project] @ embeddings

        embeddings_1 = propagation(embeddings,coarse_laplacian[num_project],q)
        embeddings_2 = propagation_community(embeddings,coarse_laplacian[num_project],communities[num_project],projections[num_project],q)
        embeddings = embeddings_1 * 0.9 + embeddings_2 * 0.1

        print("embeddings:",np.shape(embeddings),np.shape(embeddings_all[i+1]))
        embeddings = 0.4*embeddings + 0.6*embeddings_all[i+1]

        print("embeddings:",np.shape(embeddings))
        filter_    = smooth_filter(coarse_laplacian[num_project], lda)

        if power or i == 0:
            embeddings = filter_ @ (filter_ @ embeddings)

    return embeddings

def refinement_double(G,levels, projections, coarse_laplacian, embeddings_all, lda, power,q):
    embeddings = embeddings_all[0]#embeddings_all[levels]
    for i in reversed(range(levels)):
        print(embeddings.shape[0],projections[i].shape[0])
        embeddings = projections[i] @ embeddings
        print("project",embeddings,"direct",embeddings_all[i])
        embeddings = 1*embeddings #+ 0.2*embeddings_all[i]

        if power or i == 0:
            embeddings = propagation(embeddings,coarse_laplacian[i],q)

def refinement(G,levels, projections, coarse_laplacian, embeddings, lda, power,q,propa,communities,propacomm):
    for i in reversed(range(levels)):
        print("basic refine level"+str(i))
        print(embeddings.shape,projections[i].shape)
        embeddings = projections[i] @ embeddings
        filter_    = smooth_filter(coarse_laplacian[i], lda)
        if propa:   
            print("！! neiber")
            embeddings = propagation(embeddings,coarse_laplacian[i],q)
        if propacomm:
            print("！! communities")
            embeddings = propagation_community(embeddings,coarse_laplacian[i],communities[i],projections[i],q)

        if power or i == 0:
            embeddings = filter_ @ (filter_ @ embeddings)
    return embeddings

def getReduceGraph(G,G_name):
    print("G转换成邻接矩阵的形式:")
    A = np.array(nx.adjacency_matrix(G).todense())
    print('遍历邻接矩阵，获得邻接矩阵表示')
    print('创建数据集在dataset-adjacent目录下:')
    with open('./dataset-adjacent/our_reduce_graph/{}.txt'.format(G_name),'w+') as f:
        for i in range(len(A)):
            for j in range(len(A[0])):
                if A[i][j]!=0:
                    f.write('{} {} {}\n'.format(i,j,A[i][j]))
    print('创建约减后的数据集完毕....')

def loadMetisGraph(dataName):
    t=0
    rows=[]
    cols=[]
    datas=dict()
    shape=None
    max=0
    with open('./dataset/ours/{}_network.txt'.format(dataName),'r') as f:
        lines=f.readlines()
        for line in lines:
            line=line[:-1]
            datalist=line.split(' ')
            datalist[0]=int(eval(datalist[0]))
            datalist[1]=int(eval(datalist[1]))
            datalist[2]=eval(datalist[2])
            if(datalist[0]>max):
                max = datalist[0]
            if(datalist[1]>max):
                max = datalist[1]
            if(datalist[0] not in datas):
                datas[datalist[0]] = [datalist[1]]
            else:
                datas[datalist[0]].append(datalist[1])
            if(datalist[1] not in datas):
                datas[datalist[1]] = [datalist[0]]
            else:
                datas[datalist[1]].append(datalist[0])
    datalist=[]
    last =0
    for i in sorted(datas.keys()):
        if(i!=last and i!=last+1):
            print(i,last)
            for j in range(last+1,i):
                datalist.append([j,j])
            last = i
        else:
            last=i
        datalist.append(datas[i])
    print(len(datalist))
    return datalist

def loadOslomGraph(dataName):
    adjacency = list()
    with open('./dataset/ours/{}_network.txt'.format(dataName),'r') as f:
        lines=f.readlines()
        for line in lines:
            line=line[:-1]
            datalist=line.split(' ')
            #datalist=datalist[:-1]
            adjacency.append(tuple(datalist))
        return adjacency
def loadNoRepeatGraph(dataName):
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
            datalist[0] = id_remapper.get_int_id(datalist[0])
            datalist[1] = id_remapper.get_int_id(datalist[1])
            rows.append(datalist[0])
            cols.append(datalist[1])
            datas.append(datalist[2])
        shape=max(rows+cols)+1
    ret_csr_matrix=csr_matrix((datas, (rows, cols)), shape=(shape, shape))
    return ret_csr_matrix

def loadOurGraph(dataName):
    #相同数据集版本变化需要注释掉以下4行
    #if(os.path.isfile("dataset/ours/{}.gpickle".format(dataName))):
    #    G = nx.read_gpickle("dataset/ours/{}.gpickle".format(dataName))
    #    laplacian=mmread("dataset/ours/{}.mtx".format(dataName))
    #    return laplacian,G
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
    G = nx.from_scipy_sparse_matrix(ret_csr_matrix, edge_attribute='wgt')
    nx.write_gpickle(G, "dataset/ours/{}.gpickle".format(dataName))
    laplacian = laplacian_matrix(G, nodelist=range(len(G.nodes)),weight='wgt')
                #laplacian = nx.adjacency_matrix(G).todense()
    file = open("dataset/ours/{}.mtx".format(dataName), "wb")
    mmwrite("dataset/ours/{}.mtx".format(dataName), laplacian)
    file.close()
    return ret_csr_matrix,G

def loadOurGraphfast(dataName):
    G = nx.read_gpickle("dataset/ours/{}.gpickle".format(dataName))
    laplacian=mmread("dataset/ours/{}.mtx".format(dataName))
    return laplacian,G

def attribute2Graph(laplacian,G,feature, num_neighs=3):
    nodes_list = G.nodes().tolist()
    dim = G.number_of_nodes()
    all_rows = []
    all_cols = []
    all_data = []

    for i in nodes_list:
        row = []
        col = []
        data = []
        for j in nodes_list:
            col_ = []
            data_ = []
            dist = []
            feature1 = feature[i, :]
            feature2 = feature[j, :]
            if i != j:
                dist.append(LA.norm(feature1-feature2))
                col.append(j)
        ids_sort = np.argsort(np.asarray(dist))
        col_ind = (np.asarray(col_)[ids_sort]).tolist()[:num_neighs]
        for ind in col_ind:
            feat2 = feature[ind, :]
            data_.append(cosine_similarity(feature1, feature2))
        row += (np.repeat(i, num_neighs)).tolist()
        col += col_ind
        data += data_





    return  laplacian


def main():
    parser = ArgumentParser(description="CEMA")
    parser.add_argument("-d", "--dataset", type=str, default="cora", \
            help="input dataset")
    parser.add_argument("-v", "--level", type=int, default=2, \
            help="number of coarsening levels (only required by simple_coarsen)")
    parser.add_argument("-n", "--num_neighs", type=int, default=2, \
            help="control k-nearest neighbors in graph fusion process")
    parser.add_argument("-l", "--lda", type=float, default=0.1, \
            help="control self loop in adjacency matrix")
    parser.add_argument("-e", "--embed_path", type=str, default="embed_results/embeddings.npy", \
            help="path of embedding result")
    parser.add_argument("-f", "--fusion", default=True, action="store_false", \
            help="whether use graph fusion")
    parser.add_argument("-i", "--propa", default=True, action="store_false", \
            help="whether propagate")
    parser.add_argument("-t", "--community", default=True, action="store_true", \
            help="whether propagate")

    args = parser.parse_args()

    dataset = args.dataset
    feature_path = "dataset/{}/{}-feats.npy".format(dataset, dataset)
    fusion_input_path = "dataset/{}/{}.mtx".format(dataset, dataset)

    reduce_results = "reduction_results/{}/".format(dataset)
    if os.path.exists(reduce_results)==False:
        os.makedirs(reduce_results)
    mapping_path = "{}Mapping.mtx".format(reduce_results)





######Load Data######

    print("%%%%%% Loading Graph Data %%%%%%")

    laplacian,G = loadOurGraph(dataset)#fast(dataset)

    adjacency_list = None
    oslom_list = None
    if args.coarse == "metis":
        adjacency_list = loadMetisGraph(dataset)
    if args.coarse == "oslom":
        oslom_list =  loadOslomGraph(dataset)

    feature = np.load(feature_path)

    if args.fusion or args.embed_method == "graphsage":  
        feature = np.load(feature_path)


###### Attribute Import ######
              
    laplacian    = graph_fusion(laplacian, feature, args.num_neighs, args.mcr_dir, args.coarse,\
                       fusion_input_path, args.search_ratio, reduce_results, mapping_path, dataset)

##### Graph Reduction######                              
    print("%%%%%% Starting Graph Reduction %%%%%%")
    reduce_start = time.process_time()


    Gs, projections, laplacians, level = sim_coarse(laplacian, args.level)   
    reduce_time = time.process_time() - reduce_start
    print("laplacians:",np.shape(laplacian))

    communities = None
    if args.community:
        communities = buildcommunity(projections)                               


######Embed Reduced Graph######

    merger = projection_to_merger(projections)                                  
    cfeature = coarse_feature(projections,feature)                          
    embeddings_all = []
    embeddings = None
    models = []
    #print(len(Gs))
    print("level:"+str(level))
    embed_start = time.process_time()
    for i in range(level+1):
        if("Gs" in vars()):
            G = Gs[len(Gs)-i-1]
            print("use Gs")
        print("G nodes:",len(G.nodes()))                                        

                                    
        embed_start = time.process_time()
        from embedding import transductive_classifier
        mapping = identity(projections[0].shape[0])

        for p in projections:
                
                mapping = mapping @ p
        mapping = normalize(mapping, norm='l1', axis=1).transpose()

        feats = mapping @ feature 
        print(len(G.nodes()))
        print(cfeature[len(cfeature)-i-1].shape)
        embeddings = transductive_classifier.get_embedding(G,cfeature[len(cfeature)-i-1])
        embeddings_all.append(embeddings)

        if i==0:
            break

    embed_time = time.process_time() - embed_start

######Refinement######
    print("%%%%%% Starting Graph Refinement %%%%%%")
    refine_start = time.process_time()
    print("laplacians:",np.shape(laplacian))
    embeddings   = refinement_concate(G,level, projections, laplacians, embeddings_all, args.lda, args.power,args.qvalue,args.propa,args.community,communities)
    refine_time  = time.process_time() - refine_start
    print("laplacians:",np.shape(laplacian))
    print("finish embedding")

######Save Embeddings######
    np.save(args.embed_path, embeddings)
    from metric.metric import evaluation

    acc,nmi=evaluation(embeddings,dataset,level,args.concate,args.harp,args.propa,args.fusion,args.community)

    os.system("pause")

    lr("dataset/{}/".format(dataset), args.embed_path,level,args.propa,args.fusion,args.community)

######Report timing information######
    print("%%%%%% time %%%%%%")

    total_time = reduce_time + embed_time + refine_time
    print("Graph Fusion     Time: 0")
    print(f"Graph Reduction  Time: {reduce_time:.3f}")
    print(f"Graph Embedding  Time: {embed_time:.3f}")
    print(f"Graph Refinement Time: {refine_time:.3f}")
    print(f"Total Time = Fusion_time + Reduction_time + Embedding_time + Refinement_time = {total_time:.3f}")
    print("acc is "+str(acc))
    print("nmi is "+str(nmi))

if __name__ == "__main__":
    sys.exit(main())

