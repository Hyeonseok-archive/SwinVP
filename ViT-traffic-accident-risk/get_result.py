import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import json
import configparser
import pickle as pkl
from time import time
from datetime import datetime
import shutil
import argparse
import random
import math
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from lib.dataloader import normal_and_generate_dataset_time,get_mask
from lib.early_stop import EarlyStopping
from model.vit import ViT
from model.SwinVP import SwinVP
from lib.utils import mask_loss, compute_loss_vit, predict_and_evaluate_vit

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
parser.add_argument("--gpus", type=str,help="test program")
parser.add_argument("--test", action="store_true", help="test program")

args = parser.parse_args()
config_filename = args.config
with open(config_filename, 'r') as f:
    config = json.loads(f.read())
print(json.dumps(config, sort_keys=True, indent=4))


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

north_south_map = config['north_south_map']
west_east_map = config['west_east_map']


all_data_filename = config['all_data_filename']
mask_filename = config['mask_filename']




patience = config['patience']
delta = config['delta']

if config['seed'] is not None:
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic=True
    np.random.seed(seed)
    random.seed(seed)


train_rate = config['train_rate']
valid_rate = config['valid_rate']

recent_prior = config['recent_prior']
week_prior = config['week_prior']
one_day_period = config['one_day_period']
days_of_week = config['days_of_week']
pre_len = config['pre_len']
seq_len = recent_prior + week_prior

training_epoch = config['training_epoch']

def training(net,
            training_epoch,
            train_loader,
            val_loader,
            test_loader,
            high_test_loader,
            risk_mask,
            trainer,
            early_stop,
            device,
            scaler,
            data_type='nyc'
            ):
    mode = input('mode? :')
    global_step = 1
    res = []
    
        
    start_time = time()
    
    train_feature,target_time,gragh_feature,train_label = next(iter(train_loader))
    # train_feature,target_time,gragh_feature,train_label = train_feature.to(device),target_time.to(device),gragh_feature.to(device),train_label.to(device)

    # if mode == 'vit':
    #     result = net(train_feature[:,:,0,:,:], train_feature[:,:,1:,0,0].flatten(start_dim=1,))
    # else:
    #     result = net(train_feature)

    res.append(train_feature)

    return res

def main(config):
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    
    loaders = []
    scaler = ""
    train_data_shape = ""
    graph_feature_shape = ""
    for idx,(x,y,target_times,high_x,high_y,high_target_times,scaler) in enumerate(normal_and_generate_dataset_time(
                                    all_data_filename,
                                    train_rate=train_rate,
                                    valid_rate=valid_rate,
                                    recent_prior = recent_prior,
                                    week_prior = week_prior,
                                    one_day_period = one_day_period,
                                    days_of_week = days_of_week,
                                    pre_len = pre_len)):
        if args.test:
            x = x[:100]
            y = y[:100]
            target_times = target_times[:100]
            high_x = high_x[:100]
            high_y = high_y[:100]
            high_target_times = high_target_times[:100]

        if 'nyc' in all_data_filename:
            graph_x = x[:,:,[0,46,47],:,:].reshape((x.shape[0],x.shape[1],-1,north_south_map*west_east_map))
            high_graph_x = high_x[:,:,[0,46,47],:,:].reshape((high_x.shape[0],high_x.shape[1],-1,north_south_map*west_east_map))
        if 'chicago' in all_data_filename:
            graph_x = x[:,:,[0,39,40],:,:].reshape((x.shape[0],x.shape[1],-1,north_south_map*west_east_map))
            high_graph_x = high_x[:,:,[0,39,40],:,:].reshape((high_x.shape[0],high_x.shape[1],-1,north_south_map*west_east_map))


        print("feature:",str(x.shape),"label:",str(y.shape),"time:",str(target_times.shape),
            "high feature:",str(high_x.shape),"high label:",str(high_y.shape))
        if idx == 0:
            scaler = scaler
            train_data_shape = x.shape
            time_shape = target_times.shape
            graph_feature_shape = graph_x.shape

        loaders.append(Data.DataLoader(
            Data.TensorDataset(
                torch.from_numpy(x),
                torch.from_numpy(target_times),
                torch.from_numpy(graph_x),
                torch.from_numpy(y)
                ),
                batch_size=batch_size,
                shuffle=(idx == 0)
            ))
        
        if idx == 2:
            high_test_loader = Data.DataLoader(
                Data.TensorDataset(
                    torch.from_numpy(high_x),
                    torch.from_numpy(high_target_times),
                    torch.from_numpy(high_graph_x),
                    torch.from_numpy(high_y)
                    ),
                    batch_size=batch_size,
                    shuffle=(idx == 0)
                )
            
    train_loader, val_loader, test_loader = loaders

    md_type = input('mode type? : ')

    
    if md_type == 'swin':
        ViT_Model =SwinVP(in_shape=x.shape[1:], patch_size=5, num_classes=400, channels=7,
                dim=64, depth=6, heads=8, mlp_dim=128, data_type = config['data_type'])
        # ViT_Model.load_state_dict(torch.load('./logs/[swinvp] best_chicago_00036.pth'))
    elif md_type == 'vit':
        ViT_Model =ViT(image_size=20, patch_size=5, num_classes=400, channels=7,
            dim=64, depth=6, heads=8, mlp_dim=128, data_type = config['data_type'])
        # ViT_Model.load_state_dict(torch.load('./logs/[vit] best_chicago_00022.pth'))
    else:
        ViT_Model =SwinVP(in_shape=x.shape[1:], patch_size=5, num_classes=400, channels=7,
                dim=64, depth=6, heads=8, mlp_dim=128, mode='simvp', data_type = config['data_type'])
        # ViT_Model.load_state_dict(torch.load('./logs/[simvp] best_chicago_00029.pth'))

    #multi gpu
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!",flush=True)
        ViT_Model = nn.DataParallel(ViT_Model)
    ViT_Model.to(device)
    print(ViT_Model)

    


    trainer = optim.Adam(ViT_Model.parameters(), lr=learning_rate)
    early_stop = EarlyStopping(patience=patience,delta=delta)
    
    risk_mask = get_mask(mask_filename)

    res = training(
            ViT_Model,
            training_epoch,
            train_loader,
            val_loader,
            test_loader,
            high_test_loader,
            risk_mask,
            trainer,
            early_stop,
            device,
            scaler,
            data_type = config['data_type']
            )
    
    
    return res

if __name__ == "__main__":
    
    #python train_vit.py --config config/nyc/vit.json --gpus 0
    res = main(config)
    with open('chicagofeature.pickle', 'wb') as f:
        pkl.dump(res, f)
    
