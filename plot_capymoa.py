import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle 

def plot_confusion(y_true, y_pred, superclass_dict, min_val = 0, max_val = 1, colors="Blues", title = None, fontsize=8, figsize=(12,10), text = True):
    """
    Plot of the confusion matrix (% values)
    Inputs:
    - y_true: vector of ground truth labels
    - y_pred: vector of predicted labels
    - superclass_dict: dict with the classes
    - colors (optional): the cmap
    """
    l = len(superclass_dict)
    confusion_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    confusion_matrix_np = np.zeros((l, l))
    
    for i in range(0,l):
        row = sum(confusion_mat[i])
        for j in range(0,l):
            confusion_matrix_np[i][j] = confusion_mat[i][j]/row
    confusion_matrix_np = np.round(confusion_matrix_np, 2)
    
    plt.figure(figsize=figsize)
    plt.imshow(confusion_matrix_np, cmap=colors, vmin=min_val, vmax=max_val)
    plt.colorbar(shrink=0.8)

    if text is True:
        plt.yticks(np.arange(l), list(superclass_dict.values()), rotation=45, ha='right')
        plt.xticks(np.arange(l), list(superclass_dict.values()), rotation=45, ha='right')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    avg = (max_val+min_val)/2
    if text is True:
        for i in range(l):
            for j in range(l):
                if confusion_matrix_np[i][j] < avg:
                        col='black'
                else: col='white'
                plt.text(j, i, str(confusion_matrix_np[i, j]), ha='center', va='center', color=col, fontsize=fontsize)
    if title is not None:
        plt.title('Confusion matrix - '+title)
    plt.show()


def plot_accuracy_moa(instances, accuracy,len_exp = None, title = None, idx_concept_drift = None):
    plt.plot(instances, accuracy, marker='o')
    
    if len_exp is not None:
        plt.axvline(x=0, color='silver', linestyle='--', linewidth=0.8)
        for i in range(len(len_exp)-1):
            plt.axvline(x=len_exp[i], color='silver', linestyle='--', linewidth=0.8)    
    
    if idx_concept_drift is not None:
        if len(idx_concept_drift)>0:
            for j in range(0, len(idx_concept_drift)):
                plt.axvline(x=idx_concept_drift[j], color='firebrick', linestyle='--', linewidth=0.8)    

    plt.xlabel('# processed samples')
    plt.ylabel('Accuracy')
    if title is not None:
        plt.title(title)
    plt.show()

def plot_ft_bt_moa(ft_matrix, bt_matrix, colors="Blues", min_val = 0, max_val=1, title = None, first_exp=0):
    """
    This function plots the matrix for forward and backward transfer.
    Inputs:
    -np_matrix: np matrix with the values of accuracy over the experiences
    -colors (optional): the cmap
    """
    nexp = len(ft_matrix)
    np_matrix = np.round(ft_matrix, 1)

    for i in range(nexp):
        for j in range(i):
            np_matrix[i][j] = bt_matrix[i][j]

    np_matrix = np.round(np_matrix, 1)
    
    plt.imshow(np_matrix, cmap=colors, vmin=min_val, vmax=max_val)
    
    avg = (max_val+min_val)/2

    for i in range(nexp):
        for j in range(nexp):
            if np_matrix[i][j] < avg:
                col='black'
            else: col='white'
            plt.text(j, i, str(np_matrix[i][j]), ha='center', va='center', color=col, fontsize=8)

    plt.colorbar()
    plt.xticks(np.arange(nexp), np.arange(nexp)+2004+first_exp, rotation=45)
    plt.yticks(np.arange(nexp), np.arange(nexp)+2004+first_exp, rotation=45)
    plt.ylabel('Training experience')
    plt.xlabel('Testing experience')
    if title is not None:
        plt.title(title)
    else: plt.title('Accuracy matrix')
    plt.show()

def plot_ft_moa(np_matrix, is_forward = True, colors="Blues", min_val = 0, max_val=1, title = None, first_exp=0):
    """
    This function plots the matrix for forward and backward transfer.
    Inputs:
    -np_matrix: np matrix with the values of accuracy over the experiences
    -forward: True to plot the forward matrix, False to plot the backward matrix
    -colors (optional): the cmap
    
    """
    nexp = len(np_matrix)
    np_matrix = np.round(np_matrix, 1)
    if is_forward:
        for i in range(nexp):
            for j in range(i):
                np_matrix[i][j] = np.nan
    else:     
        for i in range(nexp):
            for j in range(i + 1, nexp):  # Upper diagonal elements are when j > i
                np_matrix[i][j] = np.nan
    
    plt.imshow(np_matrix, cmap=colors, vmin=min_val, vmax=max_val)
    
    avg = (max_val+min_val)/2
    if is_forward:
        for i in range(nexp):
            for j in range(i, nexp):
                if np_matrix[i][j] < avg:
                    col='black'
                else: col='white'
                plt.text(j, i, str(np_matrix[i][j]), ha='center', va='center', color=col, fontsize=8)
    else: 
        for i in range(nexp):
            for j in range(i+1):
                if np_matrix[i][j] < avg:
                    col='black'
                else: col='white'
                plt.text(j, i, str(np_matrix[i][j]), ha='center', va='center', color=col, fontsize=8)
                plt.text(j, i, str(np_matrix[i][j]), ha='center', va='center', color='black', fontsize=8)
    plt.colorbar()
    plt.xticks(np.arange(nexp), np.arange(nexp)+2004+first_exp, rotation=45)
    plt.yticks(np.arange(nexp), np.arange(nexp)+2004+first_exp, rotation=45)
    plt.ylabel('Training experience')
    plt.xlabel('Testing experience')
    if title is not None:
        plt.title('Forward transfer matrix - '+title)
    else: plt.title('Forward transfer matrix')
    plt.show()

def plot_compare_acc(rp_settings, models, scaler=False, superclass=None, title = None, len_exp = None, ylim=[0, 100]):
    file_path = 'C:/Users/eliga/OneDrive - Politecnico di Milano/2023-2024 primo semestre/Tesi/Code/CapyMOA/output/accuracy_vectors/'

    if superclass is not None:
        filenames =  superclass + '_' +rp_settings['type_RP'] + '_' + str(rp_settings['n_reduced']) + '_' + rp_settings['features'] + '_'
        if scaler:
            filenames =  superclass + '_' +rp_settings['type_RP'] + '_' + str(rp_settings['n_reduced']) + '_' + rp_settings['features'] + '_scaler_'
    
    if superclass is None:
        filenames =  rp_settings['type_RP'] + '_' + str(rp_settings['n_reduced']) + '_' + rp_settings['features'] + '_'
        if scaler:
            filenames =  rp_settings['type_RP'] + '_' + str(rp_settings['n_reduced']) + '_' + rp_settings['features'] + '_scaler_'
    
    label_dict = {
        'ht': 'Hoeffding tree',
        'knn': 'KNN',
        'hat': 'Hoeffding adaptive tree',
        'ob': 'Online bagging',
        'lb': 'Leveraging bagging',
        'ob_ht': 'Online bagging',
        'lb_ht': 'Leveraging bagging',
        'sgd': 'Softmax regression - SGD',
        'nb': 'Naive Bayes',
        'ozaboost': 'OzaBoost',
        'arf': 'Adaptive random forest'
    }

    for i in range(0, len(models)):
        file_to_open = file_path + filenames + models[i] + '.pkl' 
        with open(file_to_open, "rb") as file:
            result_window_pd = pickle.load(file)
        plt.plot(result_window_pd['instances'], result_window_pd['accuracy'], label=label_dict[models[i]])
    
    if len_exp is not None:
        plt.axvline(x=0, color='silver', linestyle='--', linewidth=0.8)
        for i in range(len(len_exp)-1):
            plt.axvline(x=len_exp[i], color='silver', linestyle='--', linewidth=0.8)  

    plt.xlabel('# processed samples')
    plt.ylabel('Accuracy')
    plt.ylim(ylim[0], ylim[1])
    plt.legend()
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if title is not None:
        plt.title(title)
    plt.show()

def plot_compare_acc_moco_imagenet(rp_settings, classifier, superclass=None, len_exp = None, ylim=[0, 100]):
    file_path = 'C:/Users/eliga/OneDrive - Politecnico di Milano/2023-2024 primo semestre/Tesi/Code/CapyMOA/output/accuracy_vectors/'
    if superclass is not None:
        filenames =  superclass + '_' +rp_settings['type_RP'] + '_' + str(rp_settings['n_reduced']) + '_' + 'moco' + '_'
    else:  filenames =  rp_settings['type_RP'] + '_' + str(rp_settings['n_reduced']) + '_' + 'moco' + '_'
    label_dict = {
        'ht': 'Hoeffding tree',
        'knn': 'KNN',
        'hat': 'Hoeffding adaptive tree',
        'ob': 'Online bagging',
        'lb': 'Leveraging bagging',
        'ob_ht': 'Online bagging',
        'lb_ht': 'Leveraging bagging',
        'sgd': 'Softmax regression - SGD',
        'nb': 'Naive Bayes',
        'ozaboost': 'OzaBoost',
        'arf': 'Adaptive random forest'
    }

    file_to_open = file_path + filenames + classifier + '.pkl' 
    with open(file_to_open, "rb") as file:
        result_window_pd_moco = pickle.load(file)
    
    plt.plot(result_window_pd_moco['instances'], result_window_pd_moco['accuracy'], label='MoCo - bucket 0')
    
    if len_exp is not None:
        plt.axvline(x=0, color='silver', linestyle='--', linewidth=0.8)
        for i in range(len(len_exp)-1):
            plt.axvline(x=len_exp[i], color='silver', linestyle='--', linewidth=0.8)  
    
    if superclass is not None:
        filenames =  superclass + '_' +rp_settings['type_RP'] + '_' + str(rp_settings['n_reduced']) + '_' + 'imagenet' + '_'
    else:  filenames =  rp_settings['type_RP'] + '_' + str(rp_settings['n_reduced']) + '_' + 'imagenet' + '_'

    file_to_open = file_path + filenames + classifier + '.pkl' 
    with open(file_to_open, "rb") as file:
        result_window_pd_imagenet = pickle.load(file)
    
    plt.plot(result_window_pd_imagenet['instances'], result_window_pd_imagenet['accuracy'], label='ResNet50 - ImageNet')
    
    plt.xlabel('# processed samples')
    plt.ylabel('Accuracy')
    plt.ylim(ylim[0], ylim[1])
    plt.legend()
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(label_dict[classifier])
    plt.show()