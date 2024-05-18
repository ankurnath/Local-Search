from cim_optimizer.solve_Ising import *
from src.envs.utils import GraphDataset
from cim_optimizer.CIM_helper import brute_force
from cim_optimizer.solve_Ising import *
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import networkx as nx

from cim_optimizer.optimal_params import maxcut_100_params
from cim_optimizer.optimal_params import maxcut_200_params
from cim_optimizer.optimal_params import maxcut_500_params

from multiprocessing.pool import Pool

def compute_cut(matrix,spins):
  return (1/4) * np.sum( np.multiply(matrix, 1 - np.outer(spins,spins)))

def load_pickle(file_path):
  with open(file_path, 'rb') as f:
      data = pickle.load(f)
  return data


def solve(graph):
    # CAC hyperparameters
    p= 0.9
    alpha= 1.1
    beta= 0.35
    gamma= 0.0005
    delta= 15
    mu= 0.7
    rho= 1
    tau= 200
    noise = 0.00

    # additional run information
    num_trials = 100
    # time_span = 25000
    time_span = 1200
    nsub = 0.02
    num_parallel_runs=20

    result = Ising(-graph).solve(num_runs = num_trials, 
                                num_timesteps_per_run = time_span, 
                                num_parallel_runs=num_parallel_runs, 
                                return_spin_trajectories_all_runs=False,
                                amplitude_control_scheme=False,
                                cac_time_step=nsub, 
                                cac_r=p, 
                                cac_alpha=alpha, 
                                cac_beta=beta, 
                                cac_gamma=gamma, 
                                cac_delta=delta,
                                cac_mu=mu,
                                cac_rho=rho,
                                cac_tau=tau,
                                suppress_statements = True,
                                hyperparameters_autotune = True,
                                hyperparameters_randomtune = False,
                                use_GPU = True,
                                use_CAC =True)
    
    spins=result.result['lowest_energy_spin_config']
    cut= (1/4) * np.sum( np.multiply(graph, 1 - np.outer(spins, spins) ) )
    return cut

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distribution', type=str,default="WattsStrogatz_200vertices_unweighted",  help='Distribution of dataset')
    args = parser.parse_args()
    arguments = []
    dataset=GraphDataset(folder_path=f'../data/testing/{args.distribution}',ordered=True)
    print ('Number of test graphs:',len(dataset))

    try:
        OPT = load_pickle(f'../data/testing/{args.distribution}/optimal', add_data_path=False)['OPT']
    except:
        OPT= None

    # results={"Cut":[]}
    cuts=[]

    # for _ in range(len(dataset)):
    for _ in range(10):
        arguments.append((dataset.get(),))

    with Pool() as pool:
        cuts=pool.starmap(solve, arguments)
    print(cuts)

    if OPT:
        for i,best_cut in enumerate(cuts):
            if best_cut>OPT.iloc[i]:
                OPT.iloc[i]=best_cut
        OPT=pd.DataFrame(OPT)
        OPT.to_pickle(f'../data/testing/{args.distribution}/optimal')
    else:
        df={'OPT':cuts}
        df=pd.DataFrame(df)
        df.to_pickle(f'../data/testing/{args.distribution}/optimal')
        print(df)


    # main()