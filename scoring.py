from __future__ import print_function
import json
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from networkx.readwrite import json_graph

def run_regression(train_embeds, train_labels, test_embeds, test_labels,dataset,embed_method,coarse_method,level,concate,harp,propa,fusion,community):
    if(fusion):
        fusion = "fusion"
    else:
        fusion = "nofusion"
    if(community):
        community = "community"
    else:
        community= "basic"
    log         = LogisticRegression(solver='lbfgs', multi_class='auto')
    log.fit(train_embeds, train_labels)
    pred_labels = (log.predict(test_embeds)).tolist()
    acc         = accuracy_score(test_labels, pred_labels)
    print("Test Accuracy: ", acc)
    logacc(embed_method,dataset+"_"+community+"_"+fusion+"_"+str(level),coarse_method,acc)

def logacc(log_name,file_name,coarse_method,acc):
    with open('./logs/' + log_name + '/'+coarse_method+ '/' + file_name + '.txt', 'a') as f:
        f.write('\n' + 'single label acc: ' + str(acc) + '\n')

def lr(dataset_dir, data_dir, dataset,embed_method,coarse_method,level,concate,harp,propa,fusion,community):
    print("%%%%%% Starting Evaluation %%%%%%")
    print("Loading data...")
    G      = json_graph.node_link_graph(json.load(open(dataset_dir + "/{}-G.json".format(dataset))))
    labels = json.load(open(dataset_dir + "/{}-class_map.json".format(dataset)))

    train_ids    = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
    test_ids     = [n for n in G.nodes() if G.node[n]['test']]
    test_ids     = test_ids[:1000]
    train_labels = [labels[str(i)] for i in train_ids]
    test_labels  = [labels[str(i)] for i in test_ids]

    embeds       = np.load(data_dir)
    train_embeds = embeds[[id for id in train_ids]]
    test_embeds  = embeds[[id for id in test_ids]]
    print("Running regression..")
    #10-90
    run_regression(train_embeds, train_labels, test_embeds, test_labels,dataset,embed_method,coarse_method,level,concate,harp,propa,fusion,community)

