import os
import sys
from sklearn.model_selection import train_test_split
current_dir = os.getcwd() # current directory
parent_dir = os.path.dirname(current_dir) # parent directory
sys.path.append(parent_dir)

from proj_function_np import gaussian_random_matrix, sparse_random_matrix_np
import pandas as pd
import numpy as np
import time
#import pickle

def create_dataset_RP(data_dir, save_dir, n_original=2048, n_reduced=100, seed=2024, type_RP='sparse', nexp=11, split_data = False, first_exp=0):

    start_time = time.time()
    
    # generate random projection matrix (numpy matrix)
    if type_RP=='sparse':
        proj_matrix = sparse_random_matrix_np(n_reduced=n_reduced, n_original=n_original, seed = seed, very_sparse=False)   
    elif type_RP=='very sparse':
        proj_matrix = sparse_random_matrix_np(n_reduced=n_reduced, n_original=n_original, seed = seed, very_sparse=True)   
    elif type_RP=='gaussian':
        proj_matrix = gaussian_random_matrix(n_reduced=n_reduced, n_original=n_original, seed = seed)   
    else: 
        raise ValueError('Type of random projection not supported, please enter sparse, very sparse or gaussian')
    
    proj_matrix = proj_matrix.T
    if not os.path.exists(save_dir +'/all'):
        os.makedirs(save_dir +'/all')
    if split_data:
        if not os.path.exists(save_dir +'/train'):
            os.makedirs(save_dir +'/train')
        if not os.path.exists(save_dir +'/test'):
            os.makedirs(save_dir +'/test')
    
    for exp in range(first_exp, nexp):
        # Import current experience
        dataset= pd.read_pickle(data_dir + '/' + str(exp)+ '.pkl')
        dataset_np = np.array(dataset.iloc[:,:-1])
        targets = dataset.iloc[:,-1:]
        proj_dataset_np = []
        for i in range(len(dataset_np)):
            proj_row = np.matmul(dataset_np[i], proj_matrix)
            proj_dataset_np.append(proj_row)

        #proj_dataset_np = np.matmul(dataset_np, proj_matrix)
        cols = ['Attr_' + str(i) for i in range(1, n_reduced + 1)]
        proj_dataset = pd.DataFrame(proj_dataset_np, columns=cols)
        proj_dataset['Target'] = targets
        proj_dataset.to_pickle(save_dir +'/all/'+ str(exp) + '.pkl', compression='infer', protocol=5, storage_options=None)
        

        if split_data:
            train_df, test_df = train_test_split(proj_dataset, test_size=0.3, random_state=2024)
            train_df.to_pickle(save_dir +'/train/'+ str(exp) + '.pkl', compression='infer', protocol=5, storage_options=None)
            test_df.to_pickle(save_dir+'/test/' + str(exp) + '.pkl', compression='infer', protocol=5, storage_options=None)

   
    end_time = time.time()
    time_to_run = (end_time-start_time)/60
    return time_to_run
    

