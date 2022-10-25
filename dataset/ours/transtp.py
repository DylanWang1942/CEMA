from argparse import Namespace
import oslom
def transTP():
    partition = list()
    t=0
    time=0
    with open('C:\\Users\\mi\\Documents\\OSLOM2\\OSLOM2\\cora_network_tab.txt','r') as f:
        lines=f.readlines()
        for line in lines:
            t+=1
            line=line[:-1]
            datalist=line.split('\t')
            #datalist=datalist[:-1]
            partition.append(tuple(datalist))
        return partition

print(transTP())