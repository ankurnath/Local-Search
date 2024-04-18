from  numba import njit
import numpy as np
import os
from src.envs.utils import GraphDataset
import pandas as pd
import pickle
from multiprocessing.pool import Pool


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
    # return best_score,None
    return best_score

from argparse import ArgumentParser

if __name__ == '__main__':
    # print('hello world')

    parser = ArgumentParser()

    parser.add_argument("--train_distribution", type=str, help="Distribution of train distribution")
    parser.add_argument("--test_distribution", type=str, help="Distribution of test distribution")
    parser.add_argument("--num_repeat", type=int,default=50, help="Distribution of dataset")
    
    args = parser.parse_args()

    try:
        gamma = pickle.load(open(f'pretrained_agents/{args.train_distribution}_heuristics/gamma', 'rb'))
    except FileNotFoundError:
        if args.train_distribution.startswith('torodial'):
            gamma=105
        else:
            gamma=50
        print('Loaded default tabu tenure')

    test_dataset=GraphDataset(f'../data/testing/{args.test_distribution}',ordered=True)
    tabu_cuts=[]

    for i in range(len(test_dataset)):
        graph=test_dataset.get()
        g=flatten_graph(graph)

        arguments=[]
        for _ in range(args.num_repeat):
            spins= np.random.randint(2, size=graph.shape[0])
            arguments.append((g,spins,gamma,graph.shape[0]))

        with Pool() as pool:
            cuts=pool.starmap(tabu, arguments)

        best_tabu_cut=np.max(cuts)
        
       

    
        tabu_cuts.append(best_tabu_cut)


    tabu_cuts=np.array(tabu_cuts)




    results={'cut':tabu_cuts}
    results=pd.DataFrame(results)

    save_folder=f'generalization/{args.train_distribution}_TS'
    os.makedirs(save_folder,exist_ok=True)
    
    for res, label in zip([results],
                          [f"results_{args.test_distribution}"]):
        save_path = os.path.join(save_folder, label)
        res.to_pickle(save_path)
        print("{} saved to {}".format(label, save_path))

    print(results['cut'].tolist())
    




    

















    



