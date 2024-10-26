import os
import pandas as pd
import time
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def retrieve_model(model_type):
    
    if model_type=='imagenet':
        weights = ResNet50_Weights.DEFAULT # weights trained on Imagenet
        feat_extract = resnet50(weights=weights)
        feat_extract.eval()
        modules = list(feat_extract.children())[:-1]
        model = nn.Sequential(*modules)
        for p in model.parameters(): # no fine tuning
            p.requires_grad = False
        model.eval()

    elif model_type=='moco':
        state_dict_path = 'state_dict.pth.tar'
        #def load_model(state_dict_path):
        model = resnet50(pretrained=False)
        model.fc = torch.nn.Identity()
        state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
    elif model_type=='moco_ft':
        state_dict_path = 'finetuned_model_moco9.pth.tar'
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(2048, 100)
        state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.fc = torch.nn.Identity()
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
    else: raise ValueError("Please provide a valid model (imagenet, moco or moco_ft)")
    return model


def extract_from_dataloader(dataloader, model):
    features = []
    targets = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for img, label in dataloader:
        img = img.to(device)
        label = label.to(device)
        
        with torch.no_grad():  
            feat = model(img)
        
        feat = torch.flatten(feat)  
    
        features.append(feat.squeeze().cpu().tolist())
        targets.append(label.cpu().numpy()[0]) 
    
    return features, targets


def extract_features(cnn_model, seed, data_dir_train, data_dir_test, save_dir, transform, first_exp=0, nexp=11):
    model = retrieve_model(model_type=cnn_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    begin = time.time()
    for exp in range(first_exp, nexp):    
        # Extract features from the training set
        dataset_train = datasets.ImageFolder(data_dir_train + '/' + str(exp), transform=transform)
        torch.manual_seed(seed)
        g = torch.Generator()
        g.manual_seed(seed)
        dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=0, generator=g)
        features, targets= extract_from_dataloader(dataloader_train, model)
        features = np.array(features)
        targets = np.array(targets)
        cols = ['Attr_' + str(i) for i in range(1, features.shape[1] + 1)]
        df_train = pd.DataFrame(features, columns=cols)
        df_train['Target'] = targets
        

        # Extract features from the test set
        dataset_test = datasets.ImageFolder(data_dir_test + '/' + str(exp), transform=transform)
        torch.manual_seed(seed)
        g = torch.Generator()
        g.manual_seed(seed)
        dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=0, generator=g)
        features, targets = extract_from_dataloader(dataloader_test, model)
        features = np.array(features)
        targets = np.array(targets)
        cols = ['Attr_' + str(i) for i in range(1, features.shape[1] + 1)]
        df_test = pd.DataFrame(features, columns=cols)
        df_test['Target'] = targets
        
        save_dir_all = save_dir+ '/all/'        
        if not os.path.exists(save_dir_all):
            os.makedirs(save_dir_all)
        all_data = pd.concat([df_train, df_test], axis=0, ignore_index=True)
        all_data.to_pickle(save_dir_all + str(exp) + '.pkl', compression='infer', protocol=5, storage_options=None)

    end = time.time()
    cnn_time = (end-begin)/60
    return cnn_time
