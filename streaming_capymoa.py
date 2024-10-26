import pandas as pd
import numpy as np
import time
import pickle
import os
from capymoa.misc import save_model, load_model
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from capymoa.base import SKClassifier

from capymoa.classifier import AdaptiveRandomForestClassifier, OnlineAdwinBagging, HoeffdingAdaptiveTree, AdaptiveRandomForestClassifier, NaiveBayes, OnlineBagging, HoeffdingTree, LeveragingBagging, KNN, SGDClassifier, OzaBoost
from capymoa.evaluation import ClassificationWindowedEvaluator, ClassificationEvaluator
from capymoa.drift.detectors import ADWIN
from capymoa.stream import NumpyStream
from moa.classifiers.trees import ARFHoeffdingTree
from plot_capymoa import plot_accuracy_moa, plot_confusion, plot_ft_moa, plot_ft_bt_moa
from capymoa.instance import LabeledInstance
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


class capymoa_classifier:
    def __init__(self, classifier, rp_settings, data_dir, nexp=11, window_size=None, superclass_dict=None, model_name = None, seed = 2024, ensemble_size=10, 
                 drift_detector = None, neighbors_knn =11, window_knn = None, first_exp=0, dir_save_models = None):
        self.classifier = classifier    #string with the name of the classifier
        self.rp_settings = rp_settings
        if rp_settings['n_reduced'] is None:
            raise ValueError("rp_settings does not have n_reduced")
        if rp_settings['type_RP'] is None:
            raise ValueError("rp_settings does not have type_RP")
        if rp_settings['features'] is None:
            raise ValueError("rp_settings does not have features")
        
        self.data_dir = data_dir

        self.nexp = nexp
        self.first_exp = first_exp
        self.len_exp = []
        self.window_size = window_size
        self.superclass_dict = superclass_dict
        self.model_name = model_name
        self.result_window = None
        self.result_window_pd = None
        self.result_cumulative = None
        self.result_cumulative_pd = None
        self.seed = seed
        self.ensemble_size = ensemble_size
        self.window_evaluator = None
        self.cumulative_evaluator = None
        self.learners_exp = [None for _ in range(nexp-first_exp)]
        self.ground_truth = []
        self.predicted = []
        self.time_ft = 0
        self.time_classifier = 0
        self.time_bt = 0
        self.ft_matrix = None
        self.bt_matrix = None
        self.ft_acc = 0
        self.bt_acc = 0
        self.in_domain_acc = 0
        self.next_domain_acc=0
        self.confusion_matrix = None
        self.mapped = False
        self.drift_detector = drift_detector                    # CapyMOA concept drift detector
        if drift_detector is not None:
            self.indices_drifts = []                                # indices that store detected drifts
            self.indices_warnings = []
        self.neighbors_knn = neighbors_knn
        self.window_knn = window_knn
        self.dir_save_models = dir_save_models


    def run_model(self, show_accuracy=True, show_confusion = False, compute_ft = True, show_ft = True, compute_bt =False):
        #print(self.data_dir)
        data_stream = self.create_stream()
        schema = data_stream.get_schema()
        if self.window_size==None:
            self.window_size = int(round(self.len_exp[0]/4, 0))

        if self.classifier == 'arf':
            learner = AdaptiveRandomForestClassifier(schema= schema, 
                                                random_seed=self.seed, ensemble_size=self.ensemble_size)
        elif self.classifier=='hat':
            learner =  HoeffdingAdaptiveTree(schema = schema, random_seed = self.seed, grace_period=100)
        elif self.classifier=='ht':
            learner = HoeffdingTree(schema=schema, random_seed = self.seed, grace_period=100)
        elif self.classifier=='nb':
            learner = NaiveBayes(schema = schema, random_seed = self.seed)
        elif self.classifier=='ob_ht':
            learner = OnlineBagging(schema=schema, random_seed=self.seed, ensemble_size=self.ensemble_size, number_of_jobs=1, 
                                    minibatch_size=1)
        #elif self.classifier=='ob_adwin':
        #    learner = OnlineAdwinBagging(schema=schema, random_seed=self.seed, ensemble_size=self.ensemble_size, minibatch_size=1, number_of_jobs=1)  
        #elif self.classifier=='lb':
        #    learner = LeveragingBagging(schema=schema, random_seed=self.seed, ensemble_size=self.ensemble_size, number_of_jobs=1, base_learner=NaiveBayes(schema=schema),
        #                minibatch_size=1)
        elif self.classifier=='lb_ht':
            learner = LeveragingBagging(schema=schema, random_seed=self.seed, ensemble_size=self.ensemble_size, number_of_jobs=1, 
                        minibatch_size=1)
        elif self.classifier=='knn':
            if self.window_knn is None:
                learner = KNN(schema=schema, random_seed = self.seed, k=self.neighbors_knn, window_size = 1000)
            else: learner = KNN(schema=schema, random_seed = self.seed, k=self.neighbors_knn, window_size = self.window_knn)
        elif self.classifier=='knn2':
            learner = KNN(schema=schema, random_seed = self.seed, k=self.neighbors_knn, window_size = 4000)
        elif self.classifier=='sgd':
            learner = SKClassifier(schema=schema, sklearner=linear_model.SGDClassifier(loss='log_loss', random_state=self.seed), random_seed = self.seed)
        elif self.classifier=='sgd2':
            learner = SGDClassifier(schema=schema, loss='log_loss', random_seed=self.seed)
        elif self.classifier=='ozaboost':
            learner = OzaBoost(schema=schema, random_seed = self.seed, boosting_iterations = 10, use_pure_boost = False) #False -> it uses Poisson 
        elif self.classifier=='oza':
            learner = OzaBoost(schema=schema, random_seed = self.seed, boosting_iterations = 10, use_pure_boost = False) #False -> it uses Poisson 
        
        else:
            raise ValueError("The classifier is not supported")
        


        start_time = time.time()

        # The window_size in ClassificationWindowedEvaluator specifies the amount of instances used per evaluation
        self.window_evaluator = ClassificationWindowedEvaluator(schema=schema, window_size=self.window_size)
        # The window_size ClassificationEvaluator just specifies the frequency at which the cumulative metrics are stored
        self.cumulative_evaluator = ClassificationEvaluator(schema=schema, window_size=self.window_size)

        k=0
        idx_len_exp = 0
        while data_stream.has_more_instances():
            class_error = 1                         # correctly classified
            instance = data_stream.next_instance()
    
            # Test
            prediction = learner.predict(instance)

            # Update metrics
            self.window_evaluator.update(instance.y_index, prediction)
            self.cumulative_evaluator.update(instance.y_index, prediction)
            if prediction is not None:
                self.ground_truth.append(instance.y_index)
                self.predicted.append(prediction)
            if instance.y_index!=prediction:
                class_error=0        
            
            # Train
            learner.train(instance)
            
            if self.drift_detector is not None:
                self.drift_detector.add_element(class_error)
                if self.drift_detector.detected_change():
                    self.indices_drifts.append(k)
                if self.drift_detector.detected_warning():
                    self.indices_warnings.append(k)

            k=k+1
            if k==self.len_exp[idx_len_exp]:
                if self.dir_save_models is None:
                    model_name = 'models/rp' + '_' + self.rp_settings['type_RP'] + '_' + str(self.rp_settings['n_reduced']) + '_' + self.rp_settings['features'] +'_' + self.classifier + '_exp_' + str(idx_len_exp) + '.pkl'
                    if self.superclass is not None:
                        model_name = 'models/' + self.superclass + '_rp' + '_' + self.rp_settings['type_RP'] + '_' + str(self.rp_settings['n_reduced']) + '_' + self.rp_settings['features'] +'_' + self.classifier + '_exp_' + str(idx_len_exp) + '.pkl'
                
                else: 
                    model_name = self.dir_save_models +  '/exp_' + str(idx_len_exp) + '.pkl'
                    self.learners_exp[idx_len_exp] = model_name
                    #if not os.path.exists(model_name):
                    #    os.makedirs(model_name)
                    save_model(learner, model_name) 

                idx_len_exp = idx_len_exp + 1
        
        end_time = time.time()
        self.result_window = self.window_evaluator.metrics_per_window() 
        self.result_cumulative = self.cumulative_evaluator.metrics_per_window()
        self.result_cumulative_pd = pd.DataFrame(self.result_cumulative)
        
        self.time_classifier =  end_time-start_time
        print(f"{self.classifier}, {self.rp_settings['features']} {self.rp_settings['type_RP']}, k={self.rp_settings['n_reduced']}")
        print(f"Cumulative accuracy: {round(self.result_cumulative_pd['accuracy'].iloc[-1], 2)}%")
        print(f"Execution time: {round(self.time_classifier/60, 2)} minutes")

        self.confusion_matrix = confusion_matrix(y_true=self.ground_truth, y_pred=self.predicted)
        self.result_window_pd = pd.DataFrame(self.result_window)
        if show_accuracy:
           
            if self.drift_detector is not None:
                plot_accuracy_moa(instances=self.result_window_pd['instances'], accuracy=self.result_window_pd['accuracy'], len_exp=self.len_exp, title = self.model_name,
                                  idx_concept_drift=self.indices_drifts)
            else: 
                plot_accuracy_moa(instances=self.result_window_pd['instances'], accuracy=self.result_window_pd['accuracy'], len_exp=self.len_exp, title = self.model_name)
        
        if show_confusion:
            if self.superclass is None:
                plot_confusion(y_true=self.ground_truth, y_pred=self.predicted,
                    superclass_dict=self.superclass_dict, min_val = 0, max_val = 1, colors="Blues", title = None, fontsize=8, figsize=(12,12),
                    text=False)
            else:
                plot_confusion(y_true=self.ground_truth, y_pred=self.predicted,
                    superclass_dict=self.superclass_dict, min_val = 0, max_val = 1, colors="Blues", title = self.superclass, fontsize=8, figsize=(8,8),
                    text=True)
        
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

        
        if show_ft and compute_ft:
            plot_ft_moa(self.ft_matrix, min_val=0, max_val=100, title = self.model_name, first_exp=self.first_exp)
        
        #if compute_ft and compute_bt:
        #    plot_ft_bt_moa(ft_matrix=self.ft_matrix, bt_matrix=self.bt_matrix, min_val=0, max_val=100, first_exp=self.first_exp)


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
        if self.mapped == False:
            self.class_indices = []
            self.class_labels = []
            self.class_names = []
            j=0
            for key in self.superclass_dict.keys():
                self.class_indices.append(j)
                self.class_labels.append(key)
                self.class_names.append(self.superclass_dict[key])
                j=j+1
            self.mapping_dict = dict(zip(self.class_labels, self.class_indices))
            self.mapped=True
        
        for jj in range(0, len(target_vec)):
            target_vec[jj] = self.mapping_dict[target_vec[jj]]
        return(target_vec)



    def accuracy_single_exp(self, one_exp, learn_exp, dir_learner, window_size=5000):
        """
        This function computes the accuracy of a model on an experience.
        Inputs:
        - one_exp: experience on which the model is tested
        - dir_learner: the name of the pickle file with the trained capyMOA learner
        """
        learner = load_model(dir_learner)
        dataset = pd.read_pickle(self.data_dir + '/' + str(one_exp)+ '.pkl')
        features = np.array(dataset.iloc[:,:-1])
        targets = np.array(dataset['Target'])
        targets = self.map_classes(target_vec=targets)
        data_stream = NumpyStream(X=features, y=targets, target_type='categorical')  
        
        evaluator = ClassificationEvaluator(schema=data_stream.get_schema(), window_size=window_size)

        
        while data_stream.has_more_instances():
            instance = data_stream.next_instance()
            prediction = learner.predict(instance)     
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
        for i in range(0, self.nexp-self.first_exp-1):
            samples_per_exp.append(self.len_exp[i+1] - self.len_exp[i])

        begin_time = time.time()
        forward_acc = np.zeros((self.nexp-self.first_exp, self.nexp-self.first_exp))
        
        for curr_exp in range(0, self.nexp-self.first_exp): # training experience
            if is_forward:
                loop_range = range(curr_exp, self.nexp-self.first_exp)
            else: loop_range = range(0, curr_exp-self.first_exp+1)
            for exp in loop_range: #testing experience
                forward_acc[curr_exp][exp] = self.accuracy_single_exp(one_exp=exp, dir_learner=self.learners_exp[curr_exp], learn_exp=curr_exp,
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
        for i in range(0, nrows):
            for j in range(0, i):
                sum_elements += self.bt_matrix[i][j]
                num_elements +=1
        self.bt_acc = sum_elements/num_elements



def save_results(classifier_capymoa, folder_save, filename_save):    
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





class capymoa_classifier_augmentation_type1:
    def __init__(self, classifier, rp_settings, data_dir_original, data_dir_augment, nexp=11, window_size=None, superclass_dict=None, 
                 model_name = None, seed = 2024, ensemble_size=10, neighbors_knn =11, window_knn = None, first_exp=0, dir_save_models = None):
        self.classifier = classifier    #string with the name of the classifier
        self.rp_settings = rp_settings
        if rp_settings['n_reduced'] is None:
            raise ValueError("rp_settings does not have n_reduced")
        if rp_settings['type_RP'] is None:
            raise ValueError("rp_settings does not have type_RP")
        if rp_settings['features'] is None:
            raise ValueError("rp_settings does not have features")
        
        self.data_dir_original = data_dir_original
        self.data_dir_augment = data_dir_augment
        self.nexp = nexp
        self.first_exp = first_exp
        self.len_exp = []
        self.window_size = window_size
        self.superclass_dict = superclass_dict
        self.model_name = model_name
        self.result_window = None
        self.result_window_pd = None
        self.result_cumulative = None
        self.result_cumulative_pd = None
        self.seed = seed
        self.ensemble_size = ensemble_size
        self.window_evaluator = None
        self.cumulative_evaluator = None
        self.learners_exp = [None for _ in range(nexp-first_exp)]
        self.ground_truth = []
        self.predicted = []
        self.time_ft = 0
        self.time_classifier = 0
        self.time_bt = 0
        self.ft_matrix = None
        self.ft_acc = 0
        self.bt_acc = 0
        self.in_domain_acc=0
        self.next_domain_acc=0
        self.confusion_matrix = None
        self.mapped = False
        self.neighbors_knn = neighbors_knn
        self.window_knn = window_knn
        self.dir_save_models = dir_save_models


    def run_model(self, show_accuracy=True, show_confusion = False, compute_ft = True, show_ft = True, compute_bt =False):
        data_stream_original = self.create_stream_original()
        schema = data_stream_original.get_schema()

        data_stream_augmented = self.create_stream_augmented()
        
        if self.window_size==None:
            self.window_size = int(round(self.len_exp[0]/4, 0))

        if self.classifier == 'arf':
            learner = AdaptiveRandomForestClassifier(schema= schema, 
                                                random_seed=self.seed, ensemble_size=self.ensemble_size)
        elif self.classifier=='hat':
            learner =  HoeffdingAdaptiveTree(schema = schema, random_seed = self.seed, grace_period=100)
        elif self.classifier=='ht':
            learner = HoeffdingTree(schema=schema, random_seed = self.seed, grace_period=100)
        elif self.classifier=='nb':
            learner = NaiveBayes(schema = schema, random_seed = self.seed)
        elif self.classifier=='ob_ht':
            learner = OnlineBagging(schema=schema, random_seed=self.seed, ensemble_size=self.ensemble_size, number_of_jobs=1, 
                                    minibatch_size=1)
        elif self.classifier=='ob_adwin':
            learner = OnlineAdwinBagging(schema=schema, random_seed=self.seed, ensemble_size=self.ensemble_size, minibatch_size=1, number_of_jobs=1)
           
        elif self.classifier=='lb':
            learner = LeveragingBagging(schema=schema, random_seed=self.seed, ensemble_size=self.ensemble_size, number_of_jobs=1, base_learner=NaiveBayes(schema=schema),
                        minibatch_size=1)
        elif self.classifier=='lb_ht':
            learner = LeveragingBagging(schema=schema, random_seed=self.seed, ensemble_size=self.ensemble_size, number_of_jobs=1, 
                        minibatch_size=1)
        elif self.classifier=='knn':
            if self.window_knn is None:
                learner = KNN(schema=schema, random_seed = self.seed, k=self.neighbors_knn, window_size = self.window_size)
            else: learner = KNN(schema=schema, random_seed = self.seed, k=self.neighbors_knn, window_size = self.window_knn)
        elif self.classifier=='sgd':
            learner = SKClassifier(schema=schema, sklearner=linear_model.SGDClassifier(loss='log_loss', random_state=self.seed), random_seed = self.seed)
        elif self.classifier=='sgd2':
            learner = SGDClassifier(schema=schema, loss='log_loss', random_seed=self.seed)
        elif self.classifier=='sgd_new':
            learner = SKClassifier(schema=schema, sklearner=linear_model.SGDClassifier(loss='log_loss', random_state=self.seed, penalty=None,
                                                                                       learning_rate='constant', eta0=0.01),
                                    random_seed = self.seed, 
                                   )
        
        elif self.classifier=='ozaboost':
            learner = OzaBoost(schema=schema, random_seed = self.seed, boosting_iterations = 10, use_pure_boost = False) #False -> it uses Poisson 
        else:
            raise ValueError("The classifier is not supported")
        


        start_time = time.time()

        # The window_size in ClassificationWindowedEvaluator specifies the amount of instances used per evaluation
        self.window_evaluator = ClassificationWindowedEvaluator(schema=schema, window_size=self.window_size)
        # The window_size ClassificationEvaluator just specifies the frequency at which the cumulative metrics are stored
        self.cumulative_evaluator = ClassificationEvaluator(schema=schema, window_size=self.window_size)

        k=0
        idx_len_exp = 0
        while data_stream_original.has_more_instances() and data_stream_augmented:
            instance_original = data_stream_original.next_instance()
            instance_augmented = data_stream_augmented.next_instance()
    
            # Test
            prediction = learner.predict(instance_original)

            # Update metrics
            self.window_evaluator.update(instance_original.y_index, prediction)
            self.cumulative_evaluator.update(instance_original.y_index, prediction)
            if prediction is not None:
                self.ground_truth.append(instance_original.y_index)
                self.predicted.append(prediction)
  
            # Train
            learner.train(instance_augmented)

            k=k+1
            if k==self.len_exp[idx_len_exp]:

                model_name = self.dir_save_models +  '/type1_aug_exp_' + str(idx_len_exp) + '.pkl'
                self.learners_exp[idx_len_exp] = model_name
                    #if not os.path.exists(model_name):
                    #    os.makedirs(model_name)
                save_model(learner, model_name) 

                idx_len_exp = idx_len_exp + 1
        
        end_time = time.time()
        self.result_window = self.window_evaluator.metrics_per_window() 
        self.result_cumulative = self.cumulative_evaluator.metrics_per_window()
        self.result_cumulative_pd = pd.DataFrame(self.result_cumulative)
        
        self.time_classifier =  end_time-start_time
        print(f"{self.classifier}, {self.rp_settings['features']} {self.rp_settings['type_RP']}, k={self.rp_settings['n_reduced']}")
        print(f"Cumulative accuracy: {round(self.result_cumulative_pd['accuracy'].iloc[-1], 2)}%")
        print(f"Execution time: {round(self.time_classifier/60, 2)} minutes")

        self.confusion_matrix = confusion_matrix(y_true=self.ground_truth, y_pred=self.predicted)
        self.result_window_pd = pd.DataFrame(self.result_window)
        if show_accuracy:
            
            if self.drift_detector is not None:
                plot_accuracy_moa(instances=self.result_window_pd['instances'], accuracy=self.result_window_pd['accuracy'], len_exp=self.len_exp, title = self.model_name,
                                  idx_concept_drift=self.indices_drifts)
            else: 
                plot_accuracy_moa(instances=self.result_window_pd['instances'], accuracy=self.result_window_pd['accuracy'], len_exp=self.len_exp, title = self.model_name)
        
        if show_confusion:
            if self.superclass is None:
                plot_confusion(y_true=self.ground_truth, y_pred=self.predicted,
                    superclass_dict=self.superclass_dict, min_val = 0, max_val = 1, colors="Blues", title = None, fontsize=8, figsize=(12,12),
                    text=False)
            else:
                plot_confusion(y_true=self.ground_truth, y_pred=self.predicted,
                    superclass_dict=self.superclass_dict, min_val = 0, max_val = 1, colors="Blues", title = self.superclass, fontsize=8, figsize=(8,8),
                    text=True)
        
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

        
        if show_ft and compute_ft:
            plot_ft_moa(self.ft_matrix, min_val=0, max_val=100, title = self.model_name, first_exp=self.first_exp)
        
        #if compute_ft and compute_bt:
        #    plot_ft_bt_moa(ft_matrix=self.ft_matrix, bt_matrix=self.bt_matrix, min_val=0, max_val=100, first_exp=self.first_exp)


    def create_stream_original(self):
        df_tot = pd.DataFrame()
        k=0
        for exp in range(self.first_exp, self.nexp):
            dataset = pd.read_pickle(self.data_dir_original + '/' + str(exp)+ '.pkl')
            k += dataset.shape[0]
            self.len_exp.append(k)
            df_tot = pd.concat([df_tot, dataset], axis=0) 
        features = np.array(df_tot.iloc[:,:-1])
        targets = np.array(df_tot['Target'])
        targets = self.map_classes(target_vec=targets)
        data_stream = NumpyStream(X=features, y=targets, target_type='categorical') 
        return(data_stream)
    
    def create_stream_augmented(self):
        df_tot = pd.DataFrame()
        k=0
        for exp in range(self.first_exp, self.nexp):
            dataset = pd.read_pickle(self.data_dir_augmented + '/' + str(exp)+ '.pkl')
            df_tot = pd.concat([df_tot, dataset], axis=0) 
        features = np.array(df_tot.iloc[:,:-1])
        targets = np.array(df_tot['Target'])
        targets = self.map_classes(target_vec=targets)
        data_stream = NumpyStream(X=features, y=targets, target_type='categorical') 
        return(data_stream)
    
    def map_classes(self, target_vec):
        if self.mapped == False:
            self.class_indices = []
            self.class_labels = []
            self.class_names = []
            j=0
            for key in self.superclass_dict.keys():
                self.class_indices.append(j)
                self.class_labels.append(key)
                self.class_names.append(self.superclass_dict[key])
                j=j+1
            self.mapping_dict = dict(zip(self.class_labels, self.class_indices))
            self.mapped=True
        
        for jj in range(0, len(target_vec)):
            target_vec[jj] = self.mapping_dict[target_vec[jj]]
        return(target_vec)



    def accuracy_single_exp(self, one_exp, learn_exp, dir_learner, window_size=5000):
        """
        This function computes the accuracy of a model on an experience.
        Inputs:
        - one_exp: experience on which the model is tested
        - dir_learner: the name of the pickle file with the trained capyMOA learner
        """
        learner = load_model(dir_learner)
        dataset = pd.read_pickle(self.data_dir_original + '/' + str(one_exp)+ '.pkl')
        features = np.array(dataset.iloc[:,:-1])
        targets = np.array(dataset['Target'])
        targets = self.map_classes(target_vec=targets)
        data_stream = NumpyStream(X=features, y=targets, target_type='categorical')  
        
        evaluator = ClassificationEvaluator(schema=data_stream.get_schema(), window_size=window_size)

        
        while data_stream.has_more_instances():
            instance = data_stream.next_instance()
            prediction = learner.predict(instance)     
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
        for i in range(0, self.nexp-self.first_exp-1):
            samples_per_exp.append(self.len_exp[i+1] - self.len_exp[i])

        begin_time = time.time()
        forward_acc = np.zeros((self.nexp-self.first_exp, self.nexp-self.first_exp))
        
        for curr_exp in range(0, self.nexp-self.first_exp): # training experience
            if is_forward:
                loop_range = range(curr_exp, self.nexp-self.first_exp)
            else: loop_range = range(0, curr_exp-self.first_exp+1)
            for exp in loop_range: #testing experience
                forward_acc[curr_exp][exp] = self.accuracy_single_exp(one_exp=exp, dir_learner=self.learners_exp[curr_exp], learn_exp=curr_exp,
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
        for i in range(0, nrows):
            for j in range(0, i):
                sum_elements += self.bt_matrix[i][j]
                num_elements +=1
        self.bt_acc = sum_elements/num_elements



class capymoa_classifier_augmentation_type3:
    def __init__(self, classifier, rp_settings, data_dir, dir_save_models, nexp=11, window_size=None, superclass_dict=None, model_name = None, seed = 2024, ensemble_size=10, 
                neighbors_knn =11, window_knn = None, first_exp=0):
        self.classifier = classifier    #string with the name of the classifier
        self.rp_settings = rp_settings
        if rp_settings['n_reduced'] is None:
            raise ValueError("rp_settings does not have n_reduced")
        if rp_settings['type_RP'] is None:
            raise ValueError("rp_settings does not have type_RP")
        if rp_settings['features'] is None:
            raise ValueError("rp_settings does not have features")
        
        self.data_dir = data_dir

        self.nexp = nexp
        self.first_exp = first_exp
        self.len_exp = []
        self.window_size = window_size
        self.superclass_dict = superclass_dict
        self.model_name = model_name
        self.result_window = None
        self.result_window_pd = None
        self.result_cumulative = None
        self.result_cumulative_pd = None
        self.seed = seed
        self.ensemble_size = ensemble_size
        self.window_evaluator = None
        self.cumulative_evaluator = None
        self.learners_exp = [None for _ in range(nexp-first_exp)]
        self.ground_truth = []
        self.predicted = []
        self.time_ft = 0
        self.time_classifier = 0
        self.time_bt = 0
        self.ft_matrix = None
        self.bt_matrix = None
        self.ft_acc = 0
        self.bt_acc = 0
        self.in_domain_acc = 0
        self.next_domain_acc=0
        self.confusion_matrix = None
        self.mapped = False
        self.neighbors_knn = neighbors_knn
        self.window_knn = window_knn
        self.dir_save_models = dir_save_models


    def run_model(self, show_accuracy=True, show_confusion = False, compute_ft = True, show_ft = True, compute_bt =False):
        data_stream, augmented = self.create_stream()
        schema = data_stream.get_schema()
        if self.window_size==None:
            self.window_size = int(round(self.len_exp[0]/4, 0))

        if self.classifier == 'arf':
            learner = AdaptiveRandomForestClassifier(schema= schema, 
                                                random_seed=self.seed, ensemble_size=self.ensemble_size)
        elif self.classifier=='hat':
            learner =  HoeffdingAdaptiveTree(schema = schema, random_seed = self.seed, grace_period=100)
        elif self.classifier=='ht':
            learner = HoeffdingTree(schema=schema, random_seed = self.seed, grace_period=100)
        elif self.classifier=='nb':
            learner = NaiveBayes(schema = schema, random_seed = self.seed)
        elif self.classifier=='ob_ht':
            learner = OnlineBagging(schema=schema, random_seed=self.seed, ensemble_size=self.ensemble_size, number_of_jobs=1, 
                                    minibatch_size=1)
        elif self.classifier=='ob_adwin':
            learner = OnlineAdwinBagging(schema=schema, random_seed=self.seed, ensemble_size=self.ensemble_size, minibatch_size=1, number_of_jobs=1)
           
        elif self.classifier=='lb':
            learner = LeveragingBagging(schema=schema, random_seed=self.seed, ensemble_size=self.ensemble_size, number_of_jobs=1, base_learner=NaiveBayes(schema=schema),
                        minibatch_size=1)
        elif self.classifier=='lb_ht':
            learner = LeveragingBagging(schema=schema, random_seed=self.seed, ensemble_size=self.ensemble_size, number_of_jobs=1, 
                        minibatch_size=1)
        elif self.classifier=='knn':
            if self.window_knn is None:
                learner = KNN(schema=schema, random_seed = self.seed, k=self.neighbors_knn, window_size = self.window_size)
            else: learner = KNN(schema=schema, random_seed = self.seed, k=self.neighbors_knn, window_size = self.window_knn)
        elif self.classifier=='sgd':
            learner = SKClassifier(schema=schema, sklearner=linear_model.SGDClassifier(loss='log_loss', random_state=self.seed), random_seed = self.seed)
        elif self.classifier=='sgd2':
            learner = SGDClassifier(schema=schema, loss='log_loss', random_seed=self.seed)
        elif self.classifier=='sgd_new':
            learner = SKClassifier(schema=schema, sklearner=linear_model.SGDClassifier(loss='log_loss', random_state=self.seed, penalty=None,
                                                                                       learning_rate='constant', eta0=0.01),
                                    random_seed = self.seed, 
                                   )
        
        elif self.classifier=='ozaboost':
            learner = OzaBoost(schema=schema, random_seed = self.seed, boosting_iterations = 10, use_pure_boost = False) #False -> it uses Poisson 
        else:
            raise ValueError("The classifier is not supported")
        


        start_time = time.time()

        # The window_size in ClassificationWindowedEvaluator specifies the amount of instances used per evaluation
        self.window_evaluator = ClassificationWindowedEvaluator(schema=schema, window_size=self.window_size)
        # The window_size ClassificationEvaluator just specifies the frequency at which the cumulative metrics are stored
        self.cumulative_evaluator = ClassificationEvaluator(schema=schema, window_size=self.window_size)

        k=0
        idx_len_exp = 0
        while data_stream.has_more_instances():
            instance = data_stream.next_instance()
    
            # Test
            if augmented[k]==0: # no augmentation
                prediction = learner.predict(instance)

            # Update metrics
                self.window_evaluator.update(instance.y_index, prediction)
                self.cumulative_evaluator.update(instance.y_index, prediction)
                if prediction is not None:
                    self.ground_truth.append(instance.y_index)
                    self.predicted.append(prediction)
    
            # Train
            learner.train(instance)

            k=k+1
            if k==self.len_exp[idx_len_exp]:
                model_name = self.dir_save_models +  '/type3aug_exp_' + str(idx_len_exp) + '.pkl'
                self.learners_exp[idx_len_exp] = model_name
                #if not os.path.exists(model_name):
                #    os.makedirs(model_name)
                save_model(learner, model_name) 

                idx_len_exp = idx_len_exp + 1
        
        end_time = time.time()
        self.result_window = self.window_evaluator.metrics_per_window() 
        self.result_cumulative = self.cumulative_evaluator.metrics_per_window()
        self.result_cumulative_pd = pd.DataFrame(self.result_cumulative)
        
        self.time_classifier =  end_time-start_time
        print(f"{self.classifier}, {self.rp_settings['features']} {self.rp_settings['type_RP']}, k={self.rp_settings['n_reduced']}")
        print(f"Cumulative accuracy: {round(self.result_cumulative_pd['accuracy'].iloc[-1], 2)}%")
        print(f"Execution time: {round(self.time_classifier/60, 2)} minutes")

        self.confusion_matrix = confusion_matrix(y_true=self.ground_truth, y_pred=self.predicted)
        self.result_window_pd = pd.DataFrame(self.result_window)
        if show_accuracy:
            
            if self.drift_detector is not None:
                plot_accuracy_moa(instances=self.result_window_pd['instances'], accuracy=self.result_window_pd['accuracy'], len_exp=self.len_exp, title = self.model_name,
                                  idx_concept_drift=self.indices_drifts)
            else: 
                plot_accuracy_moa(instances=self.result_window_pd['instances'], accuracy=self.result_window_pd['accuracy'], len_exp=self.len_exp, title = self.model_name)
        
        if show_confusion:
            if self.superclass is None:
                plot_confusion(y_true=self.ground_truth, y_pred=self.predicted,
                    superclass_dict=self.superclass_dict, min_val = 0, max_val = 1, colors="Blues", title = None, fontsize=8, figsize=(12,12),
                    text=False)
            else:
                plot_confusion(y_true=self.ground_truth, y_pred=self.predicted,
                    superclass_dict=self.superclass_dict, min_val = 0, max_val = 1, colors="Blues", title = self.superclass, fontsize=8, figsize=(8,8),
                    text=True)
        
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

        
        if show_ft and compute_ft:
            plot_ft_moa(self.ft_matrix, min_val=0, max_val=100, title = self.model_name, first_exp=self.first_exp)
        
        #if compute_ft and compute_bt:
        #    plot_ft_bt_moa(ft_matrix=self.ft_matrix, bt_matrix=self.bt_matrix, min_val=0, max_val=100, first_exp=self.first_exp)


    def create_stream(self):
        df_tot = pd.DataFrame()
        k=0
        for exp in range(self.first_exp, self.nexp):
            dataset = pd.read_pickle(self.data_dir + '/' + str(exp)+ '.pkl')
            k += dataset.shape[0]
            self.len_exp.append(k)
            df_tot = pd.concat([df_tot, dataset], axis=0) 
        features = np.array(df_tot.iloc[:,:-2])
        targets = np.array(df_tot['Target'])
        targets = self.map_classes(target_vec=targets)
        augmented = df_tot['Augmented']
        augmented = np.array(augmented)
        data_stream = NumpyStream(X=features, y=targets, target_type='categorical') 
        return data_stream, augmented
    
    def map_classes(self, target_vec):
        if self.mapped == False:
            self.class_indices = []
            self.class_labels = []
            self.class_names = []
            j=0
            for key in self.superclass_dict.keys():
                self.class_indices.append(j)
                self.class_labels.append(key)
                self.class_names.append(self.superclass_dict[key])
                j=j+1
            self.mapping_dict = dict(zip(self.class_labels, self.class_indices))
            self.mapped=True
        
        for jj in range(0, len(target_vec)):
            target_vec[jj] = self.mapping_dict[target_vec[jj]]
        return(target_vec)



    def accuracy_single_exp(self, one_exp, learn_exp, dir_learner, window_size=5000):
        """
        This function computes the accuracy of a model on an experience.
        Inputs:
        - one_exp: experience on which the model is tested
        - dir_learner: the name of the pickle file with the trained capyMOA learner
        """
        learner = load_model(dir_learner)
        dataset = pd.read_pickle(self.data_dir + '/' + str(one_exp)+ '.pkl')
        features = np.array(dataset.iloc[:,:-1])
        targets = np.array(dataset['Target'])
        targets = self.map_classes(target_vec=targets)
        data_stream = NumpyStream(X=features, y=targets, target_type='categorical')  
        
        evaluator = ClassificationEvaluator(schema=data_stream.get_schema(), window_size=window_size)

        
        while data_stream.has_more_instances():
            instance = data_stream.next_instance()
            prediction = learner.predict(instance)     
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
        for i in range(0, self.nexp-self.first_exp-1):
            samples_per_exp.append(self.len_exp[i+1] - self.len_exp[i])

        begin_time = time.time()
        forward_acc = np.zeros((self.nexp-self.first_exp, self.nexp-self.first_exp))
        
        for curr_exp in range(0, self.nexp-self.first_exp): # training experience
            if is_forward:
                loop_range = range(curr_exp, self.nexp-self.first_exp)
            else: loop_range = range(0, curr_exp-self.first_exp+1)
            for exp in loop_range: #testing experience
                forward_acc[curr_exp][exp] = self.accuracy_single_exp(one_exp=exp, dir_learner=self.learners_exp[curr_exp], learn_exp=curr_exp,
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
        for i in range(0, nrows):
            for j in range(0, i):
                sum_elements += self.bt_matrix[i][j]
                num_elements +=1
        self.bt_acc = sum_elements/num_elements




