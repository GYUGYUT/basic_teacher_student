import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import random

from pickle import TRUE

from dataloader_dir2 import *
from getmodel  import *

from tqdm import tqdm

from paser_args import *


from pytorchtools import EarlyStopping
from train_test_module import * 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 시드 설정
seed = 42
set_seed(seed)

#DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
#hyper param

cfg = { model : "vgg11",
        "epoch" : 100,
        "lr" : 0.0001,
        "batch_size": 32,
        "shuffle" : TRUE,
        "imgsize" : [300,1024],
        "num_class" : 3,
        "numworks" : 0}

args = parse_args(cfg[model])

#model
model = getModel(args.arch, cfg["num_class"])
print("선정된 모델",args.arch)
check_GPU_STATE = 0
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    check_GPU_STATE  = 2
    model = nn.DataParallel(model,device_ids=[0,1])
else:
    check_GPU_STATE  = 1
    print("Let's use", torch.cuda.device_count(), "GPUs!")    
model.to(device)

m = nn.Softmax(dim=1)
early_stopping = EarlyStopping(cfg = cfg)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

wandb.init(
    # set the wandb project where this run will be logged
    project="korail_3class_ver1",
    # entity = "alswo7400"
    entity = "amilab"
)
wandb.run.name = str(args.arch) + "_" + str(cfg["lr"]) + "_" + str(cfg["batch_size"]) + "_" + str(cfg["imgsize"]) + "_" + str(cfg["num_class"]) 
wandb.run.save()
config = {"lr":cfg["lr"],"batch_size":cfg["batch_size"],"architecture":args.arch}
wandb.init(config = config)

wandb.watch(model,loss_fn,log="all",log_freq=10)



train_data_path = r"/home/ami3/Desktop/GYU/korail/dataset/v1/train"
val_data_path = r"/home/ami3/Desktop/GYU/korail/dataset/v1/val"
test_data_path = r"/home/ami3/Desktop/GYU/korail/dataset/v1/test"

train_data = get_loader(train_data_path, cfg["imgsize"],cfg["batch_size"],cfg["shuffle"])
val_data = get_loader(val_data_path, cfg["imgsize"],cfg["batch_size"],cfg["shuffle"])

for epoch in tqdm(range(1,cfg["epoch"]+1),"전체 진행률"):
    check = train(device,args,model,train_data,val_data,loss_fn, optimizer, epoch,m,wandb,early_stopping,cfg)
    if(early_stopping.early_stop):
        del train_data 
        del val_data
        break
    else:
        pass

test_data = get_loader(test_data_path, cfg["imgsize"],cfg["batch_size"],cfg["shuffle"])
test(device,args, model, test_data,loss_fn,m,wandb,early_stopping, check_GPU_STATE,cfg)