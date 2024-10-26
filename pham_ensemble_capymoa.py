import os
import sys
current_dir = os.getcwd() # current directory
parent_dir = os.path.dirname(current_dir) # parent directory
sys.path.append(parent_dir)
import pandas as pd
import numpy as np
import time
import pickle
from capymoa.misc import save_model, load_model
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from capymoa.base import SKClassifier

from capymoa.classifier import HoeffdingAdaptiveTree,  NaiveBayes, HoeffdingTree, KNN, SGDClassifier
from capymoa.evaluation import ClassificationWindowedEvaluator, ClassificationEvaluator
from capymoa.drift.detectors import ADWIN
from capymoa.stream import NumpyStream
from plot_capymoa import plot_accuracy_moa, plot_confusion, plot_ft_moa, plot_ft_bt_moa
from capymoa.instance import LabeledInstance
from collections import Counter
import random
from proj_function_np import gaussian_random_matrix, sparse_random_matrix_np

class PhamEnsemble:
    def __init__(self, classifier,  n_models, class_dict,rp_settings, save_dir_models, data_dir,nexp=11, first_exp=0, seed=2024, window_size=None):
        """"
        - classifier indicates the base learner
        - n_models is the number of base learners
        - class_dict is a dictionary contaning the names of the classes
        - rp_settings is a dictionary that included settings to do RP (it has keys n_reduced, type_RP and features)
        - save_dir_models is the directory in which models are saved at the end of each experience
        - data_dir is directory of the dataset of features before random projection (dim 2048)
        - nexp is the index of the last experience
        - first_exp is the index of the first experience
        - seed is the random seed for reproducibility
        - window_size is the size of the window used for window evaluator
        """
        self.n_models = n_models                            # number of classifiers of the ensemble
        self.class_dict = class_dict                        # dict with the names of the classes

        self.classifier = classifier                        # string with the name of the classifier
        self.seed = seed                                    # seed for reproducibility
        self.rp_settings = rp_settings
        self.len_exp = []
        self.window_size=window_size
        self.nexp = nexp
        self.first_exp = first_exp                          #for loops will be in range(first_exp, nexp)
        self.random_matrices = []                           # list with matrices for RP
      
        self.window_evaluator = None
        self.cumulative_evaluator = None
        self.ground_truth = []
        self.predicted = []
        self.time_classifier = 0 
        self.time_ft = 0
        self.time_bt = 0

        self.result_window = None
        self.result_window_pd = None
        self.result_cumulative = None
        self.result_cumulative_pd = None
        self.learners_exp = [None for _ in range(nexp)]

        self.class_indices = []
        self.class_labels = []
        self.class_names = []
        
  
        self.ft_matrix = None
        self.ft_acc = 0
        self.bt_acc = 0
        self.next_domain_acc=0
        self.in_domain_acc=0
        self.confusion_matrix = None
        
        if rp_settings['n_reduced'] is None:
            raise ValueError("rp_settings does not have n_reduced")
        if rp_settings['type_RP'] is None:
            raise ValueError("rp_settings does not have type_RP")
        if rp_settings['features'] is None:
            raise ValueError("rp_settings does not have features")
        
        # Generate random matrices (random projection)
        if rp_settings['type_RP'] == 'sparse':
            for i in range(n_models):     
                self.random_matrices.append(sparse_random_matrix_np(n_original=rp_settings['n_original'], n_reduced= rp_settings['n_reduced'], seed=seed+i, very_sparse=False))
        
        if rp_settings['type_RP'] == 'very_sparse':
            for i in range(n_models):     
                self.random_matrices.append(sparse_random_matrix_np(n_original=rp_settings['n_original'], n_reduced= rp_settings['n_reduced'], seed=seed+i, very_sparse=True))
        
        if rp_settings['type_RP'] == 'gaussian':
            for i in range(n_models):     
                self.random_matrices.append(gaussian_random_matrix(n_original=rp_settings['n_original'], n_reduced= rp_settings['n_reduced'], seed=seed+i))

        self.data_dir = data_dir                            # directory with the features BEFORE random projection
        self.save_dir_models = save_dir_models
        if not os.path.exists(self.save_dir_models):
            os.makedirs(self.save_dir_models)

        
        
        # create the schema
        proj_matrix = sparse_random_matrix_np(n_reduced=rp_settings['n_reduced'], n_original=rp_settings['n_original'], seed = seed, very_sparse=False)   
        proj_matrix = proj_matrix.T
        dataset= pd.read_pickle(self.data_dir + '/' + str(0)+ '.pkl')
        dataset_np = np.array(dataset.iloc[:,:-1])
        targets = dataset['Target']
        targets = self.map_classes(target_vec=np.array(targets))
        proj_dataset_np = np.matmul(dataset_np, proj_matrix)
        #cols = ['Attr_' + str(i) for i in range(1, rp_settings['n_reduced'] + 1)]
        #proj_dataset = pd.DataFrame(proj_dataset_np, columns=cols)
        #proj_dataset['Target'] = targets
        stream =  NumpyStream(X=proj_dataset_np, y=targets, target_type='categorical') 
        self.schema_after_RP = stream.get_schema()
        
        #create_dataset_RP(data_dir=self.data_dir, save_dir=self.save_dir_RP, n_original=self.rp_settings['n_original'], 
        #                        n_reduced=self.rp_settings['n_reduced'], seed=2024, type_RP=self.rp_settings['type_RP'],
        #                        nexp=1, split_data = False, first_exp=self.first_exp)
    
    def project_instance(self, matrix, instance):
        """"Given the random matrix and the capymoa labeled instance, it returns the projected instance"""
        proj_vector  = np.matmul(matrix, instance.x)
        proj_instance = LabeledInstance.from_array(schema = self.schema_after_RP, x= proj_vector, y_index=instance.y_index)
        return proj_instance

    def create_stream(self):        
        df_tot = pd.DataFrame()
        k=0
        for exp in range(self.first_exp, self.nexp):
            dataset = pd.read_pickle(self.data_dir + '/' + str(exp)+ '.pkl')
            k += dataset.shape[0]
            self.len_exp.append(k)
            df_tot = pd.concat([df_tot, dataset], axis=0)
        features = np.array(df_tot.iloc[:,:-1])
        targets = np.array(df_tot['Target'])
        targets = self.map_classes(target_vec=targets)
        data_stream = NumpyStream(X=features, y=targets, target_type='categorical') 
        return(data_stream)
    
    def map_classes(self, target_vec):
        #print(target_vec)
        self.class_indices = []
        self.class_labels = []
        self.class_names = []
        j=0
        for key in self.class_dict.keys():
            self.class_indices.append(j)
            self.class_labels.append(key)
            self.class_names.append(self.class_dict[key])
            j=j+1
        self.mapping_dict = dict(zip(self.class_labels, self.class_indices))
        for jj in range(0, len(target_vec)):
            target_vec[jj] = self.mapping_dict[target_vec[jj]]
        return(target_vec)
    

    
    def run_model(self, show_accuracy=True, compute_ft=True, compute_bt = True, show_confusion=False):
        schema = self.schema_after_RP
        print("Creating stream")
        data_stream = self.create_stream()
        print("End scheme")

        if self.window_size is None:
            self.window_size = int(round(self.len_exp[0]/4, 0))


        learners = []
        for i in range(self.n_models):
            learner = None
            if self.classifier=='hat':
                learner =  HoeffdingAdaptiveTree(schema = schema, random_seed = self.seed, grace_period = 100)
            elif self.classifier=='ht':
                learner = HoeffdingTree(schema=schema, random_seed = self.seed, grace_period=100)
            elif self.classifier=='nb':
                learner = NaiveBayes(schema = schema, random_seed = self.seed)
            elif self.classifier=='knn':
                learner = KNN(schema=schema, random_seed = self.seed, k=11, window_size = self.window_size)
            elif self.classifier=='sgd':
                learner = SKClassifier(schema=schema, sklearner=linear_model.SGDClassifier(loss='log_loss', random_state=self.seed), random_seed = self.seed)
            elif self.classifier=='sgd2':
                learner = SGDClassifier(schema=schema, loss='log_loss', random_seed=self.seed)
            else: raise ValueError("The provided classifier is not supported")
            learners.append(learner)
        


        start_time = time.time()

        # The window_size in ClassificationWindowedEvaluator specifies the amount of instances used per evaluation
        self.window_evaluator = ClassificationWindowedEvaluator(schema=schema, window_size=self.window_size)
        
        # The window_size ClassificationEvaluator just specifies the frequency at which the cumulative metrics are stored
        self.cumulative_evaluator = ClassificationEvaluator(schema=schema, window_size=self.window_size)

        k=0
        idx_len_exp = 0


        while data_stream.has_more_instances():
            instance = data_stream.next_instance()
            
            projected_instances = []                # list containing projected instances
            predictions = []                        # list containing predictions for each model of the ensemble
            
            # Do random projection
            for j in range(self.n_models):
                projected_instances.append(self.project_instance(matrix=self.random_matrices[j], instance=instance))

    
            # Test
            prediction = None
            for j in range(self.n_models):
                predictions.append(learners[j].predict(projected_instances[j]))
            
            prediction = majority_voting(predictions=predictions, seed = self.seed)


            # Update metrics
            self.window_evaluator.update(instance.y_index, prediction)
            self.cumulative_evaluator.update(instance.y_index, prediction)
            if prediction is not None:
                self.ground_truth.append(instance.y_index)
                self.predicted.append(prediction)      
            
            # Train
            for j in range(self.n_models):
                learners[j].train(projected_instances[j])

            k=k+1
            if k==self.len_exp[idx_len_exp]:
                print(f"Finished processing experience: #{idx_len_exp}")
                model_name = self.save_dir_models + '/' + 'rp' + '_' + self.rp_settings['type_RP'] + '_' + str(self.rp_settings['n_reduced']) + '_' + self.rp_settings['features'] +'_' + self.classifier + '_seed' + str(self.seed) +'_exp_' + str(idx_len_exp) + '_'
               
                self.learners_exp[idx_len_exp] = model_name
                
                for j in range(0, self.n_models):
                    save_dir = model_name + '_model_' + str(j) + '.pkl'
                    save_model(learners[j], save_dir) 
                idx_len_exp = idx_len_exp + 1
        
        end_time = time.time()
        self.result_window = self.window_evaluator.metrics_per_window() 
        self.result_cumulative = self.cumulative_evaluator.metrics_per_window()
        self.result_cumulative_pd = pd.DataFrame(self.result_cumulative)
        
        self.time_classifier =  end_time-start_time
        
        print(f"Cumulative accuracy: {round(self.result_cumulative_pd['accuracy'].iloc[-1], 2)}%")
        print(f"Execution time: {round(self.time_classifier/60, 2)} minutes")
        

        
        self.confusion_matrix = confusion_matrix(y_true=self.ground_truth, y_pred=self.predicted)
        self.result_window_pd = pd.DataFrame(self.result_window)
        if show_accuracy:
            plot_accuracy_moa(instances=self.result_window_pd['instances'], accuracy=self.result_window_pd['accuracy'], len_exp=self.len_exp)
        
        if show_confusion:
            plot_confusion(y_true=self.ground_truth, y_pred=self.predicted,
                    superclass_dict=self.class_dict, min_val = 0, max_val = 1, colors="Blues", title = None, fontsize=8, figsize=(12,12),
                    text=False)
            
        if compute_ft:
            print("Computing forward transfer")
            self.compute_ft(is_forward=True)
            print(f"Time to compute forward transfer: {round(self.time_ft/60, 2)} minutes")
            self.in_domain_accuracy()
            print(f"In domain accuracy: {round(self.in_domain_acc, 2)}%")
            self.next_domain_accuracy()
            print(f"Next domain accuracy: {round(self.next_domain_acc, 2)}%")
            self.ft_accuracy()
            print(f"Forward transfer: {round(self.ft_acc, 2)}%")

        if compute_bt:
            print("Computing backward transfer") 
            self.compute_ft(is_forward=False)
            self.bt_accuracy()
            print(f"Backward transfer: {round(self.bt_acc, 2)}%")
        
        #if compute_ft:
        #    plot_ft_moa(self.ft_matrix, min_val=0, max_val=100)
        
        #if compute_ft and compute_bt:
        #    plot_ft_bt_moa(ft_matrix=self.ft_matrix, bt_matrix=self.bt_matrix, min_val=0, max_val=100)


    def accuracy_single_exp(self, one_exp, dir_learner, window_size=5000):
        """
        This function computes the accuracy of a model on an experience.
        Inputs:
        - one_exp: experience on which the model is tested
        - dir_learner: the name of the pickle file with the trained capyMOA learner
        """
        # import the learners
        learners = []
        for j in range(0, self.n_models):
            data_dir = dir_learner + '_model_' + str(j) + '.pkl'
            learner = load_model(data_dir)
            learners.append(learner) 
        
        # import data from the testing experience
        dataset = pd.read_pickle(self.data_dir + '/' + str(one_exp)+ '.pkl')
        features = np.array(dataset.iloc[:,:-1])
        targets = np.array(dataset['Target'])
        targets = self.map_classes(target_vec=targets)
        data_stream = NumpyStream(X=features, y=targets, target_type='categorical')
            
        evaluator = ClassificationEvaluator(schema=data_stream.get_schema(), window_size=window_size)


        while data_stream.has_more_instances():
            instance = data_stream.next_instance()
            
            projected_instances = []                # list containing projected instances
            predictions = []                        # list containing predictions for each model of the ensemble
            prediction = None

            # Do random projection
            for j in range(self.n_models):
                projected_instances.append(self.project_instance(matrix=self.random_matrices[j], instance=instance))

            for j in range(self.n_models):
                predictions.append(learners[j].predict(projected_instances[j]))

            prediction = majority_voting(predictions=predictions, seed = self.seed)
            evaluator.update(y_target_index = instance.y_index, y_pred_index=prediction)
        return round(evaluator.accuracy(), 2)

    def compute_ft(self, is_forward = True):
        """It computes the matrix of forward/backward transfer accuracy 
        Inputs:
        - nexp = number of experiences
        - is_forward = bool, if true it computes forward transfer, if false it computes backward transfer
        """
        samples_per_exp = []
        samples_per_exp.append(self.len_exp[0])
        for i in range(self.first_exp, self.nexp-1):
            samples_per_exp.append(self.len_exp[i+1] - self.len_exp[i])

        begin_time = time.time()
        forward_acc = np.zeros((self.nexp, self.nexp))
        
        for curr_exp in range(self.first_exp, self.nexp): # training experience
            if is_forward:
                loop_range = range(curr_exp, self.nexp)
            else: loop_range = range(self.first_exp, curr_exp+1)
            for exp in loop_range: #testing experience
                #print(f"Testing experience {exp} on the model trained on experience {curr_exp}")
                forward_acc[curr_exp][exp] = self.accuracy_single_exp(one_exp=exp, dir_learner=self.learners_exp[curr_exp], 
                                                                       window_size=samples_per_exp[exp])
        end_time = time.time()
        if is_forward:
            self.time_ft = end_time - begin_time        
            self.ft_matrix = forward_acc  
        else:
            self.time_bt = end_time - begin_time
            self.bt_matrix = forward_acc
    
    def ft_accuracy(self):
        """It computes the forward transfer metric starting from the forward transfer matrix"""
        if self.ft_matrix is None:
            raise ValueError("The forward transfer matrix has not been computed")    
        nrows = len(self.ft_matrix)
        num_elements = 0
        sum_elements = 0
        for i in range(0, nrows-1):
            for j in range(i+1, nrows):
                sum_elements += self.ft_matrix[i][j]
                num_elements +=1
        self.ft_acc = sum_elements/num_elements

    def next_domain_accuracy(self):
        """It computes the next domain accuracy starting from the forward transfer matrix"""
        if self.ft_matrix is None:
            raise ValueError("The forward transfer matrix has not been computed")  
        nrows = len(self.ft_matrix)
        sum_nda = 0
        for i in range(0, nrows-1):
            sum_nda += self.ft_matrix[i][i+1]
        self.next_domain_acc = sum_nda/(nrows-1)
    
    def in_domain_accuracy(self):
        """It computes the in domain accuracy starting from the forward transfer matrix"""
        sum_ida = 0
        if self.ft_matrix is None:
            raise ValueError("The forward transfer matrix has not been computed")  
        nrows = len(self.ft_matrix)
        for i in range(0, nrows):
            sum_ida += self.ft_matrix[i][i]
        self.in_domain_acc = sum_ida/nrows

    def bt_accuracy(self):
        """It computes the backward transfer metric starting from the forward transfer matrix"""
        num_elements = 0
        sum_elements = 0

        if self.bt_matrix is None:
            raise ValueError("The backward transfer matrix has not been computed")    
        nrows = len(self.bt_matrix)
        for i in range(0, nrows-1):
            for j in range(0, i):
                sum_elements += self.bt_matrix[i][j]
                num_elements +=1
        self.bt_acc = sum_elements/num_elements

def majority_voting(predictions, seed):
    """"Given a list of predictions, it returns the final prediction using majority voting"""
    count = Counter(predictions)
    winner_count=max(count.values())

    possible_winners = []
    for key, value in count.items():
        if value==winner_count:
            possible_winners.append(key)
    
    if len(possible_winners)==1:                # no ties
        winner = possible_winners[0]
    else:                                           # tie-breaking rule (random choice between those with more votes)
        random.seed(seed)
        winner = possible_winners[random.randint(a=0, b=len(possible_winners)-1)]    
    return winner


def save_results_pham(classifier_capymoa, folder_save, filename_save):    
    ft_save_dir = folder_save + '/ft_matrices/' 
    bt_save_dir = folder_save + '/bt_matrices/' 
    acc_save_dir = folder_save + '/accuracy_vectors/'
    confusion_save_dir = folder_save + '/confusion_matrices/'
    len_exp_save_dir = folder_save + '/len_exp/'

    # create folders if they do not exist
    if not os.path.exists(ft_save_dir):
        os.makedirs(ft_save_dir)
    if not os.path.exists(bt_save_dir):
        os.makedirs(bt_save_dir)
    if not os.path.exists(acc_save_dir):
        os.makedirs(acc_save_dir)
    if not os.path.exists(confusion_save_dir):
        os.makedirs(confusion_save_dir)
    if not os.path.exists(len_exp_save_dir):
        os.makedirs(len_exp_save_dir)
    

    with open(ft_save_dir + filename_save +'.pkl', "wb") as file:
        pickle.dump(classifier_capymoa.ft_matrix, file)
    
    with open(bt_save_dir + filename_save +'.pkl', "wb") as file:
        pickle.dump(classifier_capymoa.bt_matrix, file)
    
    classifier_capymoa.result_window_pd.to_csv(acc_save_dir + filename_save +'.csv', index=False)

    with open(confusion_save_dir + filename_save +'.pkl', "wb") as file:
        pickle.dump(classifier_capymoa.confusion_matrix, file)
        
    with open(len_exp_save_dir + filename_save +'.pkl', "wb") as file:
        pickle.dump(classifier_capymoa.len_exp, file)