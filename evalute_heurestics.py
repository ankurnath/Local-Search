from  numba import njit
import numpy as np
import os
from src.envs.utils import GraphDataset
import pandas as pd
import pickle


@njit
def flatten_graph(graph):
    """
    Flatten a graph into matrices for adjacency, weights, start indices, and end indices.

    Parameters:
    - graph (adjacency matrix): The input graph to be flattened.

    Returns:
    - numpy.ndarray: Flattened adjacency matrix.
    - numpy.ndarray: Flattened weight matrix.
    - numpy.ndarray: Start indices for nodes in the flattened matrices.
    - numpy.ndarray: End indices for nodes in the flattened matrices.
    """
    flattened_adjacency = []
    flattened_weights = []
    num_nodes = graph.shape[0]
    
    node_start_indices = np.zeros(num_nodes,dtype=np.int64)
    node_end_indices = np.zeros(num_nodes,dtype=np.int64)
    
    for i in range(num_nodes):
        node_start_indices[i] = len(flattened_adjacency)
        for j in range(num_nodes):
            if graph[i, j] != 0:
                flattened_adjacency.append(j)
                flattened_weights.append(graph[i, j])
                
        node_end_indices[i] = len(flattened_adjacency)

    return (
        np.array(flattened_adjacency),
        np.array(flattened_weights),
        node_start_indices,
        node_end_indices
    )



@njit
def standard_greedy(graph):
    adj_matrix, weight_matrix, start_list, end_list=graph
    
    n=len(start_list)
    delta_local_cuts=np.zeros(n)
    spins=np.ones(n)
    
    
    curr_score=0
    for i in range(n):
        for j,weight in zip(adj_matrix[start_list[i]:end_list[i]],
                  weight_matrix[start_list[i]:end_list[i]]):
                
            delta_local_cuts[i]+=weight*(2*spins[i]-1)*(2*spins[j]-1)
            curr_score+=weight*(spins[i]+spins[j]-2*spins[i]*spins[j])

    curr_score/=2    
    # best_score=curr_score
    
    flag=True
    
    while flag:
        arg_gain=np.argsort(-delta_local_cuts)
        flag=False
        for v in arg_gain:
            if spins[v]:
                if delta_local_cuts[v]<0:
                    flag=False
                    break
                    
                curr_score+=delta_local_cuts[v]
                delta_local_cuts[v]=-delta_local_cuts[v]
                
                for u,weight in zip(adj_matrix[start_list[v]:end_list[v]],
                                     weight_matrix[start_list[v]:end_list[v]]):

                    delta_local_cuts[u]+=weight*(2*spins[u]-1)*(2-4*spins[v])

                spins[v] = 1-spins[v]
                flag=True
                break
                  
    return curr_score,spins



@njit
def mca(graph,spins):
    adj_matrix, weight_matrix, start_list, end_list=graph
    
    n=len(start_list)
    delta_local_cuts=np.zeros(n)
    
    
    
    curr_score=0
    for i in range(n):
        for j,weight in zip(adj_matrix[start_list[i]:end_list[i]],weight_matrix[start_list[i]:end_list[i]]):
                
            delta_local_cuts[i]+=weight*(2*spins[i]-1)*(2*spins[j]-1)
            curr_score+=weight*(spins[i]+spins[j]-2*spins[i]*spins[j])

    curr_score/=2   
    best_score=curr_score
    
    flag=True
    
    while flag:
        arg_gain=np.argsort(-delta_local_cuts)
        flag=False
        for v in arg_gain:
            
            if delta_local_cuts[v]<=0:
                flag=False
                break
                    
            curr_score+=delta_local_cuts[v]
            delta_local_cuts[v]=-delta_local_cuts[v]

            for u,weight in zip(adj_matrix[start_list[v]:end_list[v]],
                                 weight_matrix[start_list[v]:end_list[v]]):

                delta_local_cuts[u]+=weight*(2*spins[u]-1)*(2-4*spins[v])

            spins[v] =1-spins[v]
            flag=True
            break
                  
    return curr_score,spins

@njit
def tabu(graph,spins,tabu_tenure,max_steps):
    adj_matrix, weight_matrix, start_list, end_list=graph
    
    n=len(start_list)
    delta_local_cuts=np.zeros(n)
    
    tabu_list=np.ones(n)*-10000
    curr_score=0
    for i in range(n):
        for j,weight in zip(adj_matrix[start_list[i]:end_list[i]],
                  weight_matrix[start_list[i]:end_list[i]]):
                
            delta_local_cuts[i]+=weight*(2*spins[i]-1)*(2*spins[j]-1)
            curr_score+=weight*(spins[i]+spins[j]-2*spins[i]*spins[j])
            

    curr_score/=2    
    best_score=curr_score

    for t in range(max_steps):
        arg_gain=np.argsort(-delta_local_cuts)
        for v in arg_gain:
            if (t-tabu_list[v]> tabu_tenure) or (best_score < curr_score + delta_local_cuts[v]):

                tabu_list[v] = t

                curr_score+=delta_local_cuts[v]
                delta_local_cuts[v]=-delta_local_cuts[v]
                
                for u,weight in zip(adj_matrix[start_list[v]:end_list[v]],
                                     weight_matrix[start_list[v]:end_list[v]]):

                    delta_local_cuts[u]+=weight*(2*spins[u]-1)*(2-4*spins[v])

                spins[v] = 1-spins[v]

                break

                
        best_score=max(curr_score,best_score)
    return best_score,None

from argparse import ArgumentParser

if __name__ == '__main__':
    # print('hello world')

    parser = ArgumentParser()

    parser.add_argument("--distribution", type=str, help="Distribution of dataset")
    parser.add_argument("--num_repeat", type=int,default=100, help="Distribution of dataset")
    parser.add_argument("--gamma", type=int, default=20, help="Tabu Tenure")
    args = parser.parse_args()

    save_folder=f'pretrained agents/{args.distribution}_heuristics/data'
    os.makedirs(save_folder,exist_ok=True)
    print(save_folder)

    try:
        args.gamma = pickle.load(open(f'pretrained_agents/{args.distribution}_heuristics/gamma', 'rb'))
    except FileNotFoundError:
        print('Loaded default tabu tenure')

    test_dataset=GraphDataset(f'../data/testing/{args.distribution}',ordered=True)

    tabu_cuts=[]
    mca_cuts=[]
    sg_cuts=[]

    for i in range(len(test_dataset)):
        graph=test_dataset.get()
        g=flatten_graph(graph)

        best_tabu_cut=0
        best_mca_cut=0
        
        for _ in range(args.num_repeat):
            spins= np.random.randint(2, size=graph.shape[0])

            tabu_cut,_=tabu(g,spins.copy(),tabu_tenure=args.gamma,max_steps=graph.shape[0]*2)
            
            best_tabu_cut=max(best_tabu_cut,tabu_cut)

            mca_cut,_=mca(g,spins.copy())
            best_mca_cut=max(mca_cut,best_mca_cut)
        sg_cut,_=standard_greedy(g)
    
        tabu_cuts.append(best_tabu_cut)
        mca_cuts.append(best_mca_cut)
        sg_cuts.append(sg_cut)

    tabu_cuts=np.array(tabu_cuts)
    mca_cuts=np.array(mca_cuts)
    sg_cuts=np.array(sg_cuts)

    print('Ratio (Tabu/Standard Greedy):',(tabu_cuts/sg_cuts).mean())
    print('Ratio (MCA/Standard Greedy):',(mca_cuts/sg_cuts).mean())

    df={'TS':tabu_cuts,'MCA':mca_cuts,'SG':sg_cuts}
    df['Instance'] = [os.path.basename(file) for file in test_dataset.file_paths]
    df=pd.DataFrame(df)
    print(df)
    df.to_pickle(os.path.join(save_folder,'results'))

















    



