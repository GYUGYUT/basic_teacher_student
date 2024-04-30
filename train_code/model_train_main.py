import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import random
import torch.nn.functional as F
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
# knowledge distillation loss
def distillation(y, labels, teacher_scores, T=10, alpha=0.1):
    # distillation loss + classification loss
    # y: student_out
    # labels: hard label
    # teacher_scores: soft label
    return nn.KLDivLoss(reduction='batchmean')(F.log_softmax(y/T), F.softmax(teacher_scores/T)) * (T*T * 2.0 + alpha) + F.cross_entropy(y,labels) * (1.-alpha)




def run():
    #model
    model,model2 = getModel(args.arch, cfg["num_class"])

    

    # Initialize the model weights using Xavier initialization
    
    print("선정된 모델",args.arch)
    check_GPU_STATE = 0
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        check_GPU_STATE  = 2
        model = nn.DataParallel(model,device_ids=[0,1])
        model2 = nn.DataParallel(model2,device_ids=[0,1])
    else:
        check_GPU_STATE  = 1
        print("Let's use", torch.cuda.device_count(), "GPUs!")    
    model.to(device)
    model2.to(device)

    m = nn.Sigmoid()
    early_stopping = EarlyStopping(cfg = cfg)
    train_loss_fn = distillation
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model2.parameters(), lr=cfg["lr"])

    wandb.init(
        # set the wandb project where this run will be logged
        project="eyes_5class_teacher_Adam_data_aug",
        # entity = "alswo7400"
        entity = "alswo7400"
    )
    wandb.run.name = str(args.arch) + "_" + str(cfg["lr"]) + "_" + str(cfg["batch_size"]) + "_" + str(cfg["imgsize"]) + "_" + str(cfg["num_class"]) + "_" + str(cfg["train_loss_pra"]) 
    wandb.run.save()
    config = {"lr":cfg["lr"],"batch_size":cfg["batch_size"],"teacher_architecture" : model,"student_architecture" : model2}
    wandb.init(config = config)

    wandb.watch(model,loss_fn,log="all",log_freq=10)
    wandb.watch(model2,loss_fn,log="all",log_freq=10)



    train_data_path = r"/home/gpu7/eyes/data_eyes/dataset_train"
    val_data_path = r"/home/gpu7/eyes/data_eyes/dataset_val"
    test_data_path = r"/home/gpu7/eyes/data_eyes/dataset_test"

    train_data = get_loader(train_data_path, cfg["imgsize"],cfg["batch_size"],cfg["shuffle"])
    val_data = get_loader2(val_data_path, cfg["imgsize"],cfg["batch_size"],cfg["shuffle"])

    print( train_data )
    print( val_data )
    for epoch in tqdm(range(1,cfg["epoch"]+1),"전체 진행률"):
        check = train(device,args,model,model2,train_data,val_data,loss_fn,train_loss_fn, optimizer, epoch,m,wandb,early_stopping,cfg)
        if(early_stopping.early_stop):
            del train_data 
            del val_data
            break
        else:
            pass

    test_data = get_loader2(test_data_path, cfg["imgsize"],cfg["batch_size"],cfg["shuffle"])
    test(device,args, model,model2, test_data,loss_fn,m,wandb,early_stopping, check_GPU_STATE,cfg)


#hyper param
# script = ["mobilenet_teacher","resnet50_teacher_mobilenet_student"] 
# "resnet101_teacher_mobilenet_student","mobilenet_teacher","mobilenet_teacher_hardswish_dropout"
script = ["mobilenet_teacher"] 
script2 = [[1,0.001]] 
batch_sizes = [512,256,128]
for i in script:
    for j in script2:
        for select_batch in batch_sizes:
            #DEVICE
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = None
            cfg = { model : i,
                    "epoch" : 1000,
                    "lr" : 0.0001,
                    "batch_size": select_batch,
                    "shuffle" : TRUE,
                    "imgsize" : [224,224],
                    "num_class" : 5,
                    "numworks" : 16,
                    "train_loss_pra" : j,
                    "patience" : 20}
            args = parse_args(cfg[model])
            a = run()
            del a