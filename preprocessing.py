# import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import treelstm
import numpy as np
import time
import argparse
import json, sys, os
import pandas as pd 

# libraries for encoding and making graph
import dgl
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# read preprocessed training dataset
df = pd.read_json('C:/Users/Catherine/Desktop/ProgramTranslation/ai-program-translation/tree2tree_dataset/CS-JS/AS/preprocessed_progs_train.json')
# df.tail(6)

# Building trees
i =set()
for item in data["target_ast"]:
    x = str(item).replace("[","").replace("]","").replace("{","").replace("}","").replace("''","").replace(":","").replace(",","").split()
    for num,h in enumerate(x, start = 0):
        if(h == "'root'"):
            i.add(x[num + 1])
name_to_int = dict((name, number) for number, name in enumerate(i))
idx = sorted(name_to_int.values())
vectors = np.zeros((len(idx),max(idx) + 1))
vectors[np.arange(len(vectors)),idx] = 1
vector_tensor =  th.from_numpy(vectors)

tree_container = []
for num,item in enumerate(data["target_ast"], start = 0):
    edge_list = []
    tensor_list = []
    location = 1
    total = 0
    g = dgl.DGLGraph()
    input_1 = str(data["target_ast"][num]).replace("root","").replace("children","").replace("[","").replace("]","").replace("'","").replace(":","").replace(",","").replace("{","{ ").replace("}"," }").split()

    #recursive
    
    def fill():
        global edge_list
        global tensor_list
        global location
        global total
        global input_1
        
        item = input_1[location]
        index = name_to_int["'"+item+"'"]
        tensor_to_add = vector_tensor.data[index]
        g.add_nodes(1)
        tensor_list.append(tensor_to_add)
        current_node =total
        total += 1
        location += 1
        item = input_1[location]
        if item == '{':
            location += 1
            edge_list.append((total, current_node))
            fill()
            item = input_1[location]
            if item == '{':
                location += 1
                edge_list.append((total, current_node))
                fill()
                item = input_1[location]
        if item == '}':
            location += 1
            return
        
    
    fill()
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    #g.add_edges(dst, src)
    g.ndata['info'] = th.randn(total,len(name_to_int))
    for num, item in  enumerate(tensor_list, start = 0) :
        g.ndata['info'][num] = tensor_list[num]
    tree_container.append(g)
#print example graph
nx.draw_kamada_kawai(tree_container[78].to_networkx(), with_labels=True)

