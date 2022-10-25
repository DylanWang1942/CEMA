import igraph
import numpy as np
from numpy import linalg as LA
import json
import igraph
import networkx as nx
from networkx.readwrite import json_graph
from networkx.linalg.laplacianmatrix import laplacian_matrix
from scipy.io import mmwrite
from scipy.sparse import csr_matrix,coo_matrix, diags, identity, triu, tril
from itertools import combinations
import pymetis
from argparse import Namespace
import oslom
#import infomap
def cosine_similarity(x, y):
    dot_xy = abs(np.dot(x, y))
    norm_x = LA.norm(x)
    norm_y = LA.norm(y)
    if norm_x == 0 or norm_y == 0:
        if norm_x == 0 and norm_y == 0:
            similarity = 1
        else:
            similarity = 0
    else:
        similarity = dot_xy/(norm_x * norm_y)
    return similarity

def maximum (A, B):
    ## calculate max{A, B}
    BisBigger = A-B
    BisBigger.data = np.where(BisBigger.data < 0, 1, 0)
    return A - A.multiply(BisBigger) + B.multiply(BisBigger)

def feats2graph(feature, num_neighs, mapping):
    # number of nodes in fine graph
    fine_dim   = mapping.shape[1]
    # number of nodes in coarse graph
    coarse_dim = mapping.shape[0]

    all_rows   = []
    all_cols   = []
    all_data   = []
    for i in range(coarse_dim):
        row  = []
        col  = []
        data = []
        node_list = ((mapping[i,:].nonzero())[1]).tolist()
        if len(node_list)-1 > num_neighs:
            for j in node_list:
                col_  = []
                data_ = []
                dist  = []
                feat1 = feature[j, :]
                for k in node_list:
                    if j != k:
                        feat2 = feature[k, :]
                        dist.append(LA.norm(feat1-feat2))
                        col_.append(k)
                ids_sort = np.argsort(np.asarray(dist))
                col_ind  = (np.asarray(col_)[ids_sort]).tolist()[:num_neighs]
                for ind in col_ind:
                    feat2 = feature[ind, :]
                    data_.append(cosine_similarity(feat1, feat2))
                row  += (np.repeat(j, num_neighs)).tolist()
                col  += col_ind
                data += data_
        else:
            for pair in combinations(node_list, 2):
                feat1 = feature[pair[0], :]
                feat2 = feature[pair[1], :]
                row.append(pair[0])
                col.append(pair[1])
                data.append(cosine_similarity(feat1, feat2))
        all_rows += row
        all_cols += col
        all_data += data

    adj_initial      = csr_matrix((all_data, (all_rows, all_cols)), shape=(fine_dim, fine_dim))
    adj_max          = maximum(triu(adj_initial), tril(adj_initial).transpose())
    adj_final        = adj_max + adj_max.transpose()
    degree_matrix    = diags(np.squeeze(np.asarray(adj_final.sum(axis=1))), 0)
    laplacian_matrix = degree_matrix - adj_final

    return laplacian_matrix

def json2mtx(dataset):
    G_data    = json.load(open("dataset/{}/{}-G.json".format(dataset, dataset)))
    G         = json_graph.node_link_graph(G_data)
    laplacian = laplacian_matrix(G, nodelist=range(len(G.nodes)),weight='wgt')
    file = open("dataset/{}/{}.mtx".format(dataset, dataset), "wb")
    mmwrite("dataset/{}/{}.mtx".format(dataset, dataset), laplacian)
    file.close()
    file = open("dataset/ours/{}.mtx".format(dataset), "wb")
    mmwrite("dataset/ours/{}.mtx".format(dataset), laplacian)
    file.close()
    laplacian = nx.to_scipy_sparse_matrix(G,weight='wgt',format='csr')
    G = nx.from_scipy_sparse_matrix(laplacian, edge_attribute='wgt')
    return laplacian,G

def mtx2matrix(proj_name):
    data = []
    row  = []
    col  = []
    with open(proj_name) as ff:
        for i,line in enumerate(ff):
            info = line.split()
            if i == 0:
                NumReducedNodes = int(info[0])
                NumOriginNodes  = int(info[1])
            else:
                row.append(int(info[0])-1)
                col.append(int(info[1])-1)
                data.append(1)
    matrix = csr_matrix((data, (row, col)), shape=(NumReducedNodes, NumOriginNodes))
    return matrix


def mtx2graph(mtx_path):
    G = nx.Graph()
    with open(mtx_path) as ff:
        for i,line in enumerate(ff):
            info = line.split()
            if i == 0:
                num_nodes = int(info[0])
            elif int(info[0]) < int(info[1]):
                G.add_edge(int(info[0])-1, int(info[1])-1, wgt=abs(float(info[2])))

    ## add isolated nodes
    for i in range(num_nodes):
        G.add_node(i)
    return G

def read_levels(level_path):
    with open(level_path) as ff:
        levels = int(ff.readline()) - 1
    return levels

def read_time(cputime_path):
    with open(cputime_path) as ff:
        cpu_time = float(ff.readline())
    return cpu_time

def construct_proj_laplacian(laplacian, levels, proj_dir):
    coarse_laplacian = []
    projections      = []
    adjacency = diags(laplacian.diagonal(), 0) - laplacian
    G = nx.from_scipy_sparse_matrix(adjacency, edge_attribute='wgt')
    Gs=[G]
    for i in range(levels):
        projection_name = "{}/Projection_{}.mtx".format(proj_dir, i+1)
        projection      = mtx2matrix(projection_name)
        projections.append(projection.transpose())
        coarse_laplacian.append(laplacian)
        print(i)
        #if i != (levels-1):
        print(projection.shape[0],projection.shape[1])
        print(laplacian.shape[0],laplacian.shape[1])
        laplacian = projection @ (laplacian @ (projection.transpose()))
        adjacency = diags(laplacian.diagonal(), 0) - laplacian
        G = nx.from_scipy_sparse_matrix(adjacency, edge_attribute='wgt')
        Gs.append(G)
    coarse_laplacian.append(laplacian)
    return projections, coarse_laplacian, Gs

def affinity(x, y):
    dot_xy = (np.dot(x, y))**2
    norm_x = (LA.norm(x))**2
    norm_y = (LA.norm(y))**2
    return dot_xy/(norm_x*norm_y)

def smooth_filter(laplacian_matrix, lda):
    dim        = laplacian_matrix.shape[0]
    adj_matrix = diags(laplacian_matrix.diagonal(), 0) - laplacian_matrix + lda * identity(dim)
    degree_vec = adj_matrix.sum(axis=1)
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.squeeze(np.asarray(np.power(degree_vec, -0.5)))
    d_inv_sqrt[np.isinf(d_inv_sqrt)|np.isnan(d_inv_sqrt)] = 0
    degree_matrix  = diags(d_inv_sqrt, 0)
    norm_adj       = degree_matrix @ (adj_matrix @ degree_matrix)
    return norm_adj

def spec_coarsen(filter_, laplacian,thresh = 0.3):
    np.random.seed(seed=1)

    ## power of low-pass filter
    power = 2
    ## number of testing vectors
    t = 7
    ## threshold for merging nodes
    #thresh = 0.3

    adjacency = diags(laplacian.diagonal(), 0) - laplacian
    G = nx.from_scipy_sparse_matrix(adjacency)
    tv_list = []
    num_nodes = len(G.nodes())

    ## generate testing vectors in [-1,1], 
    ## and orthogonal to constant vector
    for _ in range(t):
        tv = -1 + 2 * np.random.rand(num_nodes)
        tv -= np.ones(num_nodes)*np.sum(tv)/num_nodes
        tv_list.append(tv)
    tv_feat = np.transpose(np.asarray(tv_list))

    ## smooth the testing vectors
    for _ in range(power):
        tv_feat = filter_ @ tv_feat
    matched = [False] * num_nodes
    degree_map = [0] * num_nodes

    ## hub nodes are more important than others,
    ## treat hub nodes as seeds
    for (node, val) in G.degree():
        degree_map[node] = val
    sorted_idx = np.argsort(np.asarray(degree_map))
    row = []
    col = []
    data = []
    cnt = 0
    for idx in sorted_idx:
        if matched[idx]:
            continue
        matched[idx] = True
        cluster = [idx]
        for n in G.neighbors(idx):
            if affinity(tv_feat[idx], tv_feat[n]) > thresh and not matched[n]:
                cluster.append(n)
                matched[n] = True
        row += cluster
        col += [cnt] * len(cluster)
        data += [1] * len(cluster)
        cnt += 1
    mapping = csr_matrix((data, (row, col)), shape=(num_nodes, cnt))
    coarse_laplacian = mapping.transpose() @ laplacian @ mapping
    return coarse_laplacian, mapping

def sim_coarse(laplacian, level):
    projections = []
    laplacians = []
    adjacency = diags(laplacian.diagonal(), 0) - laplacian
    G = nx.from_scipy_sparse_matrix(adjacency, edge_attribute='wgt')
    Gs = [G]
    for i in range(level):
        filter_ = smooth_filter(laplacian, 0.1)
        laplacians.append(laplacian)
        laplacian, mapping = spec_coarsen(filter_, laplacian)
        print("不同粗化等级的Laplace：",i,np.shape(laplacian))
        projections.append(mapping)

        print("Coarsening Level:", i+1)
        print("Num of nodes: ", laplacian.shape[0], "Num of edges: ", int((laplacian.nnz - laplacian.shape[0])/2))
        adjacency = diags(laplacian.diagonal(), 0) - laplacian
        G = nx.from_scipy_sparse_matrix(adjacency, edge_attribute='wgt')
        Gs.append(G)

    #adjacency = diags(laplacian.diagonal(), 0) - laplacian
    laplacians.append(laplacian)
    print("num laplacian:",len(laplacians))
    #G = nx.from_scipy_sparse_matrix(adjacency, edge_attribute='wgt')
    return Gs, projections, laplacians, level

def sim_coarse_fusion(laplacian):
    level = 1
    mapping = identity(laplacian.shape[0])
    for _ in range(level):
        filter_ = smooth_filter(laplacian, 0.1)
        laplacian, map_ = spec_coarsen(filter_, laplacian,thresh=1)
        mapping = mapping @ map_
    mapping = mapping.transpose()
    return mapping

def buildcommunity(projections):
    communities = []
    for t in projections:
        community = dict()
        project_arr = t.toarray()
        for i in range(len(project_arr)):
            for j in range(len(project_arr[i])):
                if(project_arr[i][j]==1):
                    if j not in community:
                        community[j] = [i]
                    else:
                        community[j].append(i)
        communities.append(community)
    return communities

def propagation_community(embedding,laplacian,community,projections,end):
    print("community propagation")
    adj_matrix = diags(laplacian.diagonal(), 0) - laplacian
    #print(adj_matrix)
    #print(type(embedding))
    G = nx.from_scipy_sparse_matrix(adj_matrix)
    new_embedding = embedding.copy()
    time = 0
    while(1):
        dis = 0
        for i in G.nodes():
            avg_embedding = np.zeros(embedding.shape[1])
            belong = community[np.argmax(projections[i])]
            if(len(list(belong))==1):
                continue
            #print(len(list(G.neighbors(i))))
            for j in belong:
                avg_embedding += embedding[j]
            all = len(list(belong))
            avg_embedding = avg_embedding/all
            new_embedding[i] = (embedding[i] + avg_embedding)/2
            dis += sum(np.absolute(new_embedding[i]-embedding[i]))
        #for i in range(len(embedding)):
        #    print(sum(np.absolute(new_embedding[i]-embedding[i])))
        #    dis += sum(np.absolute(new_embedding[i]-embedding[i]))
        dis = dis/len(embedding)
        time+=1
        print(dis)
        embedding = new_embedding.copy()
        if(dis<end or time>=1):
            break
    return embedding


def propagation(embedding,laplacian,end):
    print("neighbor propagation")
    adj_matrix = diags(laplacian.diagonal(), 0) - laplacian
    #print(adj_matrix)
    #print(type(embedding))
    G = nx.from_scipy_sparse_matrix(adj_matrix)
    new_embedding = embedding.copy()
    time = 0
    while(1):
        dis = 0
        for i in G.nodes():
            avg_embedding = np.zeros(embedding.shape[1])
            if(len(list(G.neighbors(i)))==0):
                continue
            #print(len(list(G.neighbors(i))))
            for j in G.neighbors(i):
                avg_embedding += embedding[j]
            all = len(list(G.neighbors(i)))
            avg_embedding = avg_embedding/all
            new_embedding[i] = (embedding[i] + avg_embedding)/2
            dis += sum(np.absolute(new_embedding[i]-embedding[i]))
        #for i in range(len(embedding)):
        #    print(sum(np.absolute(new_embedding[i]-embedding[i])))
        #    dis += sum(np.absolute(new_embedding[i]-embedding[i]))
        dis = dis/len(embedding)
        time+=1
        #print(dis)
        embedding = new_embedding.copy()
        if(dis<end or time>2):
            break
    return embedding

def info(laplacian):
    rows=[]
    cols=[]
    datas=[]
    projections = []
    laplacians = []
    im = infomap.Infomap("--two-level")
    G = nx.from_scipy_sparse_matrix(laplacian,edge_attribute='weight')
    for source, target in G.edges:
        im.add_link(source, target)
    im.run()
    communities = im.get_modules()
    size = im.num_top_modules
    for i in communities.keys():
        rows.append(i)
        cols.append(communities[i]-1)
        datas.append(1)
    mapping = csr_matrix((datas, (rows, cols)), shape=(len(G.nodes),size))
    laplacians.append(laplacian)
    laplacian = mapping.transpose() @ laplacian @ mapping
    laplacians.append(laplacian)
    #print(laplacian.shape[0])
    projections.append(mapping)

    G = nx.from_scipy_sparse_matrix(laplacian, edge_attribute='wgt')
    return G, projections, laplacians,1

def oslo(laplacian,adjacency):
    args = Namespace()
    args.min_cluster_size = 0
    args.oslom_exec = "dataset\\ours\\oslom_undir.exe"
    args.oslom_args = ["-w"]#oslom.DEF_OSLOM_ARGS
    rows=[]
    cols=[]
    datas=[]
    projections = []
    laplacians = []
    clusters = oslom.run_in_memory(args, adjacency)
    result = clusters[0]['clusters']
    rows=[]
    cols=[]
    datas=[]
    orisize = laplacian.shape[0]
    size = len(result)
    for i in result:
        for j in i['nodes']:
            rows.append(int(j['id']))
            cols.append(int(i['id']))
            datas.append(1)
    mapping = csr_matrix((datas, (rows, cols)), shape=(orisize,size))
    laplacians.append(laplacian)
    laplacian = mapping.transpose() @ laplacian @ mapping
    laplacians.append(laplacian)
    print(laplacian.shape[0])
    projections.append(mapping)

    G = nx.from_scipy_sparse_matrix(laplacian, edge_attribute='wgt')
    return G, projections, laplacians,1
def projection_to_merger(projections):
    level = len(projections)
    merger=[]
    for i in range(level):
        merge = dict()
        coodata = projections[i].tocoo()
        row = coodata.row
        col = coodata.col
        data = coodata.data
        for j in range(len(row)):
            if data[j] == 1 :
                merge[row[j]] = col[j]
        merger.append(merge)
    return merger
def metis(laplacian,adjacency_list):
    size = int(len(adjacency_list)/2)#int(len(adjacency_list)**0.5)
    n_cuts, membership = pymetis.part_graph(size, adjacency=adjacency_list)
    rows=[]
    cols=[]
    datas=[]
    projections = []
    laplacians = []
    for i in range(len(membership)):
        rows.append(i)
        cols.append(membership[i])
        datas.append(1)
    mapping = csr_matrix((datas, (rows, cols)), shape=(len(membership),size))
    laplacians.append(laplacian)
    laplacian = mapping.transpose() @ laplacian @ mapping
    laplacians.append(laplacian)
    projections.append(mapping)
    adjacency = diags(laplacian.diagonal(), 0) - laplacian
    G = nx.from_scipy_sparse_matrix(laplacian, edge_attribute='wgt')
    mmwrite("reduction_results/Gs_bgll.mtx", adjacency,field="real")
    transmtx()
    return G, projections, laplacians,1

def bgll(laplacian,level):
    #g = igraph.Graph.Weighted_Adjacency(np.array(nx.adjacency_matrix(from_scipy_sparse_matrix(laplacian)).todense()).tolist(),mode=ADJ_UNDIRECTED)
    time = 0
    sources, targets = laplacian.nonzero()
    weights = laplacian.todense()[sources, targets]
    g = igraph.Graph(zip(sources, targets), directed=False, edge_attrs={'weight': weights})
    Vertexclusters = g.community_multilevel(return_levels=True)
    maxlevel = len(Vertexclusters)
    print("Reduce Max Level:",maxlevel)
    if level>maxlevel:
        level = maxlevel
    #finalgraph = Vertexclusters[level-1].cluster_graph()
    projections = []
    laplacians = []
    adjacency = diags(laplacian.diagonal(), 0) - laplacian
    G = nx.from_scipy_sparse_matrix(adjacency, edge_attribute='wgt')
    Gs = [G]
    print("final level:",level)
    if level != 0:
        if len(Vertexclusters)>0:
            gra=firstmapping(len(Vertexclusters[0].membership))
        else:
            print(laplacian)
            print(Vertexclusters)
            print("error:no cluster")
            exit(0)
        for i,v in enumerate(Vertexclusters):
            time+=1
            laplacians.append(laplacian)
            gra,mapping = transmappingFinal(v,gra)
            laplacian = mapping.transpose() @ laplacian @ mapping
            projections.append(mapping)
            print("Coarsening Level:", i+1)
            print("Num of nodes: ", laplacian.shape[0], "Num of edges: ", int((laplacian.nnz - laplacian.shape[0])/2))
            adjacency = diags(laplacian.diagonal(), 0) - laplacian
            G = nx.from_scipy_sparse_matrix(adjacency, edge_attribute='wgt')
            Gs.append(G)
            if time == level:
                break
    adjacency = diags(laplacian.diagonal(), 0) - laplacian
    #adjacency = laplacian
    laplacians.append(laplacian)
    #G = nx.from_scipy_sparse_matrix(adjacency, edge_attribute='wgt')
    #mmwrite("reduction_results/Gs_bgll.mtx", adjacency,field="integer")
    mmwrite("reduction_results/Gs_bgll.mtx", adjacency,field="real")
    transmtx()
    return Gs, projections, laplacians, level

def transmtx():
    t = open("reduction_results/Gs_bgll_new.mtx","w")
    f = open("reduction_results/Gs_bgll.mtx","r")
    q = 0
    for i in f.readlines():
        q+=1
        if(q>3):
            newline = i.split(" ")
            wline = "\t".join(str(i) for i in newline)
            t.write(wline)
    t.close()
    f.close()


def transmapping(g,gra):
    size = len(g)
    t=0
    q=[]
    z={}
    h=np.zeros((len(gra),size))
    for (k,v) in gra.items():
        h[k][g.membership[v]] = 1 
    for idx,value in enumerate(g.membership):
        if value not in q:
            q.append(value)
            z[value]=idx
            t+=1
        if t==size:
            break
    return z,h
def transmappingFinal(g,gra):
    size = len(g)
    t=0
    q=[]
    z={}
    rows=[]
    cols=[]
    datas=[]
    for (k,v) in gra.items():
        rows.append(k)
        cols.append(g.membership[v])
        datas.append(1)
    mapping = csr_matrix((datas, (rows, cols)), shape=(len(gra),size))
    for idx,value in enumerate(g.membership):
        if value not in q:
            q.append(value)
            z[value]=idx
            t+=1
        if t==size:
            break
    return z,mapping
def transmappingTrans(g,gra):
    size = len(g)
    t=0
    q=[]
    z={}
    h=np.zeros((size,len(gra)))
    for (k,v) in gra.items():
        h[g.membership[v]][k] = 1
    for idx,value in enumerate(g.membership):
        if value not in q:
            q.append(value)
            z[value]=idx
            t+=1
        if t==size:
            break
    return z,h
def firstmapping(size):
    z={}
    for idx in range(size):
        z[idx]=idx
    return z

def coarse_feature(projections,feature):
    level = len(projections)
    merger=[feature]
    lastmerger = feature
    for i in range(level):
        merge = dict()
        mergelist = []
        count = dict()
        coodata = projections[i].tocoo()
        row = coodata.row
        col = coodata.col
        data = coodata.data
        for j in range(len(row)):
            if data[j] == 1 :
                if col[j] not in merge:
                    #print(lastmerger[row[j]])
                    merge[col[j]] = lastmerger[row[j]]#feature[row[j]]
                    count[col[j]] = 1
                else:
                    #print(lastmerger[row[j]])
                    merge[col[j]] += lastmerger[row[j]]#feature[row[j]]
                    count[col[j]] +=1
        for j in merge:
            merge[j] = merge[j]/count[j]
        key_arr = sorted(merge.keys())
        for key in key_arr:
            mergelist.append(merge[key])
        merge = coo_matrix(mergelist).todense()
        merger.append(merge)
    return merger
