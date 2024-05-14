from  numba import njit
import numpy as np
import os
from src.envs.utils import GraphDataset
import pandas as pd
import pickle
from multiprocessing.pool import Pool
import networkx as nx
from collections import defaultdict



dataset=GraphDataset(f'../data_MinDominateSet/validation/BA_800-1200',ordered=True)
# df={'Greedy':[],'Networkx':[]}
df=defaultdict(list)




for i in range(len(dataset)):
    graph=dataset.get()
    # print(graph.shape)
    # break
    graph=nx.from_numpy_array(graph)

    # covered = set([node for node in graph.nodes()])
    covered= set([])
    solution=set([])
    merginal_gain={node:graph.degree(node) for node in graph.nodes()}

    flag=True

    while flag:
        best_node=max(merginal_gain,key=merginal_gain.get)
        if merginal_gain[best_node]>0:
            solution.add(best_node)
            covered.add(best_node)
            for neighbor in graph.neighbors(best_node):
                covered.add(neighbor)
            for node in graph.nodes():
                merginal_gain[node]=0
                for neighbor in graph.neighbors(node):
                    if neighbor not in covered:
                        merginal_gain[node]+=1
        else:
            flag=False
    
    df['Greedy'].append(len(solution))
    # df['Networkx'].append(len(nx.dominating_set(G=graph)))

df=pd.DataFrame(df)
print(df)
print(df['Greedy'].mean())
# print(df['Networkx'].mean())
