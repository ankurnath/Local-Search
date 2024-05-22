from cim_optimizer.solve_Ising import *
from src.envs.utils import GraphDataset

from cim_optimizer.solve_Ising import *
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import networkx as nx
import os


from multiprocessing.pool import Pool

def compute_cut(matrix,spins):
  return (1/4) * np.sum( np.multiply(matrix, 1 - np.outer(spins,spins)))

def load_pickle(file_path):
  with open(file_path, 'rb') as f:
      data = pickle.load(f)
  return data

from cim_optimizer.optimal_params import maxcut_100_params
from cim_optimizer.optimal_params import maxcut_200_params
from cim_optimizer.optimal_params import maxcut_500_params


def solve(graph,hyperparameters):
    

    result=Ising(-graph).solve(hyperparameters_autotune=True,
                           hyperparameters_randomtune=False,
                           return_lowest_energies_found_spin_configuration=True,
                           return_lowest_energy_found_from_each_run=False,
                           return_spin_trajectories_all_runs=False,
                           return_number_of_solutions=1,
                           suppress_statements=True,
                           **hyperparameters)

    
    spins=result.result['lowest_energy_spin_config']
    cut= (1/4) * np.sum( np.multiply(graph, 1 - np.outer(spins, spins) ) )
    return cut,spins

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distribution', type=str,default="WattsStrogatz_200vertices_weighted",  help='Distribution of dataset')
    parser.add_argument('--pool', type=int,default=20,  help="Number of pools")
    parser.add_argument("--device", type=int,default=None, help="cuda device")
    parser.add_argument("--num_runs",type=int,default=100,help="Number of runs per instance")
    parser.add_argument("--time_span",type=int,default=25000,help="Time span")
    parser.add_argument("--num_parallel_runs",type=int,default=100,help="Number of parallel runs")
    parser.add_argument("--model",type=str,default="CAC", help="Number of parallel runs")

    args = parser.parse_args()

    hyperparameters=maxcut_200_params()
    hyperparameters['num_runs']=args.num_runs
    hyperparameters['num_timesteps_per_run']=  args.time_span
    hyperparameters['num_parallel_runs']=  args.num_parallel_runs


    if torch.cuda.is_available():
        hyperparameters['use_GPU']=True
        if args.device is None:
            device = 'cuda:0' 
        else:
            device=f'cuda:{args.device}'
    else:
        device='cpu'

    if args.model=='CAC':
        hyperparameters['use_CAC']=True
    elif args.model=='AHC':
        hyperparameters['use_CAC']=False
    else:
        raise ValueError("Unknown options")

        
    dataset=GraphDataset(folder_path=f'../data/testing/{args.distribution}',ordered=True)
    print ('Number of test graphs:',len(dataset))
       
    cuts=[]
    spins=[]


    for _ in range(len(dataset)):
        graph=dataset.get()
        cut,spin=solve(graph,hyperparameters)
        cuts.append(cut)
        spins.append(spin)


    df={'cut':cuts,'Solution':spins}
    df=pd.DataFrame(df)
    print(df)
    save_folder=f"pretrained agents/{args.distribution}_{args.model}/data"
    os.makedirs(save_folder,exist_ok=True)
    
    df.to_pickle(os.path.join(save_folder,'results'))

