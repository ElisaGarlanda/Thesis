import numpy as np

def gaussian_random_matrix(n_original, n_reduced, seed):
    """
    This function generates a matrix for random projection. 
    Inputs:
    - n_original: dimensionality of the data that need to be projected 
    - n_reduced: dimensionality of the reduced data
    - seed: seed for reproducibility
    
    Output: n_reduced x n_original random matrix (np matrix)
    It extracts the elements of the matrix from a gaussian with mean=0 and standard deviation=1/n_reduced
    """
    rng = np.random.default_rng(seed=seed)
    nrows = n_reduced
    ncols = n_original
    matrix = rng.normal(loc=0.0, scale=1.0, size=(nrows, ncols))
    matrix /= np.sqrt(n_reduced) # divide by k (see paper Li, Hastie, Church)
    # in this way the norm of each column has expected value 1
    return matrix

def sparse_random_matrix_np(n_original, n_reduced, seed, s=3, very_sparse=False):
    """
    This function generates a sparse matrix for random projection. To be used to compare time wrt Gaussian RP 
    Inputs:
    - n_original: dimensionality of the data that need to be projected 
    - n_reduced: dimensionality of the reduced data
    - seed: seed for reproducibility
    - s (optional): by default it is 3, as proposed by Achiloptas
    - very_sparse (optional): if true, it implements very sparse random projection
    Output: n_reduced x n_original random matrix (numpy matrix)
    With very_sparse=False, it extracts the elements of the matrix as proposed by Achiloptas with s=3
    With very_sparse=True, it implements very sparse random projection (Li et al.) with s = sqrt(n_original)
    You will have to multiply this matrix by the input vector x_red=Rx_input
    """
    rng = np.random.default_rng(seed=seed)
    nrows = n_reduced
    ncols = n_original
    if very_sparse==True:
        s = round(np.sqrt(n_original))
    values = np.array([-np.sqrt(s), 0, np.sqrt(s)])
    probs = np.array([ 1/(2*s), 1-1/s, 1/(2*s)])
    matrix_np = rng.choice(a=values, size=(nrows, ncols), p=probs)
    matrix_np /= np.sqrt(n_reduced) # divide by k (see paper Li, Hastie, Church)    
    return matrix_np


def project_vector_np(proj_matrix, vec):
    """
    This function projects a vector using a given projection matrix. 
    Inputs:
    - proj_matrix: a numpy projection matrix 
    - vec: a dict with the vector that needs to be projected
    
    Output: a dict of the projected vector
    """
    ft = np.array(list(vec.values()))
    proj_feat = np.matmul(proj_matrix, ft) # apply RP (non-sparse)
    #proj_feat_dict = dict(enumerate(proj_feat.flatten(), 0))
    proj_feat_dict = dict(enumerate(proj_feat, 0))
    return proj_feat_dict