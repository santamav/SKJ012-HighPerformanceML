#!/usr/bin/env python3

import sys, os
sys.path.append(os.getcwd())
from icecream import ic
import numpy as np
import time
from sjk012.classifiers.cnn import *
from sjk012.data_utils import get_CIFAR10_data
from sjk012.layers import *
from sjk012.solver import Solver
import pyximport

"""
1. From the jupyter lab project, develop a standalone python module that is able to execute the neural network in your computer. 
    It should report the network accuracy and its execution time.
2. Extend this implementation using mpi4py in order to distribute parts of the data to different mpi processes and perform a data parallel version of the 
    previous neural network (synchronizing the data between the different mpi processes whenever it is necessary). You can test this implementation in your own computer. 
    The mpi version should also report the network accuracy and its execution time.
    Run your code on patan with different numbers of mpi processes (1, 2, 4, 6, and 8).
3. Write a short document explaining how you have launched the different experiments on patan and describing the obtained results. 
    You should show the accuracy and the execution time obtained for different numbers of mpi processes. Both of them in numerical format and with a figure. 
"""


def prepare_dependencies():
    import os
    from icecream import ic
    
    # Get original notebook path and LD_LIBRARY_PATH
    notebook_path = ic(os.getcwd())
    original_ld_library_path = ic(os.environ.get('LD_LIBRARY_PATH'))
    
    # Software and install_prefix as defined in 01_setup.ipynb
    software_path = ic(os.path.join(notebook_path, 'sjk012', 'software'))
    install_prefix = ic(os.path.join(software_path, 'opt'))
    
    # Paths for BLIS and OpenBLAS
    blis_path = os.path.join(install_prefix, 'blis', 'lib')
    openblas_path = os.path.join(install_prefix, 'openblas', 'lib')
    
    # Print the paths
    print(f"notebook_path: {notebook_path}")
    print(f"install_prefix: {install_prefix}")
    print(f"blis_path: {blis_path}")
    print(f"openblas_path: {openblas_path}")
    
    # Verify that the paths exist
    assert os.path.exists(blis_path), f"BLIS path {blis_path} does not exist"
    assert os.path.exists(openblas_path), f"OpenBLAS path {openblas_path} does not exist"
    
    # Update the environment
    os.environ['LD_LIBRARY_PATH'] = blis_path
    os.environ['LD_LIBRARY_PATH'] += ":"
    os.environ['LD_LIBRARY_PATH'] += openblas_path
    if original_ld_library_path is not None:
        os.environ['LD_LIBRARY_PATH'] += ":"
        os.environ['LD_LIBRARY_PATH'] += original_ld_library_path
    _ = ic(os.environ['LD_LIBRARY_PATH'])
    
    pyximport.install(reload_support=True, pyimport=True)

def get_data():
    data = get_CIFAR10_data()
    print('Data loaded')
    for k, v in list(data.items()):
        print(f"{k}: {v.shape}")
        
    return {
        'X_train': data['X_train'].astype(np.float32),
        'y_train': data['y_train'].astype(np.int8),
        'X_val': data['X_val'].astype(np.float32),
        'y_val': data['y_val'].astype(np.int8),
    }

def build_model(data, num_epochs=1, batch_size=50):
    model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001)
    solver = Solver(model, 
        data,
        num_epochs=num_epochs, 
        batch_size=batch_size,
        update_rule='adam',
        optim_config={
        'learning_rate': 1e-3,},
        verbose=True, print_every=20)
    return model, solver

def show_results(solver, data):
    # Print final training accuracy.
    print(
        "Training accuracy:",
        solver.check_accuracy(data['X_train'].astype(np.float32), data['y_train'])
    )
    # Print final validation accuracy.
    print(
        "Validation accuracy:",
        solver.check_accuracy(data['X_val'].astype(np.float32), data['y_val'])
    )
    # Shoud we plot the results for the report?

def main():
    # Prepare the dependencies
    prepare_dependencies()
    # Import the data
    data = get_data()
    # Prepare the model for training
    model, solver = build_model(data)
    # Train the model
    solver.train()
    # Check the model's results
    show_results(solver, data)

if __name__ == "__main__":
    main()
    
