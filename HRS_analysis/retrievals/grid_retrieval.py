
## IMPORTS ##

from pathlib import Path

from sys import path  # add directory where starships is located
path.append('/home/mathisb/Github/starships')

import numpy as np
import matplotlib.pyplot as plt

import re
from multiprocessing import Pool
import time

from starships import retrieval
from starships.plotting_fcts import get_plot_labels
import starships.retrieval_utils as retrieval_utils
import starships.planet_obs as pl_obs

from starships import petitradtrans_utils

petitradtrans_utils.log.setLevel('INFO')
pl_obs.log.setLevel("WARNING")



## CHANGE THIS ##

yaml_path_in = Path.home() / Path('/home/mathisb/Github/HRS_analysis/retrievals/')
yaml_file_in = yaml_path_in / Path('trappist1d_grid_H2O.yaml')

param1_name = 'N2'
param1 = np.linspace(-11, 0, 100)  # N2 abundance

param2_name = 'H2O'
param2 = np.linspace(-12, 0, 100)  # greenhouse gas abundance

n_process = 8  # nombre de CPU alloués

name_save = 'd_N2_H2O_4n.npz'



## SETUP RETRIEVAL ##

retrieval.setup_retrieval(input_parameters=yaml_file_in)
n_steps, pos, wlkr_file_out, yaml_file, good_to_go = retrieval.prepare_run(yaml_file=yaml_file_in)

param1_mesh, param2_mesh = np.meshgrid(param1, param2)
param_grid = np.array([np.ravel(param1_mesh), np.ravel(param2_mesh)]).T



## RUN GRID RETRIEVAL ##

if __name__ == "__main__":
    with Pool(n_process) as pool:
        outputs = pool.map(retrieval.lnprob, param_grid)



## SAVE RESULTS ##

logl_grid_2d = np.reshape(outputs, param1_mesh.shape)  # param1_mesh and param2_mesh have the same shape

results = np.column_stack([
    param_grid[:, 0],
    param_grid[:, 1],
    outputs,
])

np.savez(
    '/home/mathisb/scratch/HRS_retrievals/grid_retrievals/' + name_save,
    param1_name       = param1_name,
    param1_input_grid = param1,
    param2_name       = param2_name,
    param2_input_grid = param2,
    param1            = param_grid[:, 0],
    param2            = param_grid[:, 1],
    logl_grid         = logl_grid_2d,
    results           = results,
)
