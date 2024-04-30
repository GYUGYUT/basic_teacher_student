import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import random
import os
from tqdm import tqdm
from activate_fun import *

from confusion_matrix import *
from report_save import *
from cal_accuracy import *

def label_make(class_num,label):
    label = [ 1 if (i == label) else 0 for i in range(class_num) ]
    return torch.Tensor(label)

def train(device,args,model,model2,train_data,val_data,loss_fn,train_loss_fn, optimizer, epoch,m,wandb,early_stopping,cfg):
    model.eval()
    model2.train()
    loss = 0.0
    total_loss = 0
    train_acc = 0.0
    for batch_id, (X, y) in enumerate(tqdm(train_data," %d  Train!!"% epoch)):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        with torch.no_grad():
            pred = model(X)
        pred2 = model2(X)
        loss = train_loss_fn(torch.softmax(pred,dim=1),y,torch.softmax(pred2,dim=1))

        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc += calc_accuracy(pred2, y)
        
    wandb.log({"train_acc": (train_acc / (batch_id+1))*100, "train_loss": total_loss / len(train_data)},step=epoch)    
    print("epoch {} batch id {} loss {:.6f} train acc {:.6f}%".format(epoch, batch_id+1, (total_loss / len(train_data)), (train_acc/ (batch_id+1))*100))
    if(epoch%10 == 0): #pt저장
        save_path_name = str(args.arch) + "_" + str(cfg["lr"]) + "_" + str(cfg["batch_size"]) + "_" + str(cfg["imgsize"]) + "_" + str(cfg["num_class"]) + "_" + str(cfg["train_loss_pra"]) 
        path = os.path.join(args.save_path, 'One_GPU_Version_{}_train_{}.pt'.format(save_path_name,epoch))
        path2 = os.path.join(args.save_path2, 'Two_GPU_Version_{}_train_{}.pt'.format(save_path_name,epoch))
        torch.save(model2.state_dict(),path2)
        torch.save(model2.module.state_dict(), path)
    
    val_acc = 0.0
    val_loss = 0.0
    model.eval()
    model2.eval()
    val_total_loss = 0.0
    for batch_id, (X, y) in enumerate(tqdm(val_data," %d val!!!!"% epoch)):
        example_image = []
        X, y = X.to(device), y.to(device)

        with torch.no_grad():
            model.training = False
            model2.training = False
            # Compute prediction error
            pred = model(X)
            pred2 = model2(X)
            val_loss = loss_fn(pred2, y)
            val_total_loss += val_loss.item()

        val_acc += calc_accuracy(pred2, y)
        _, max_indices = torch.max(pred2, 1)
        
        example_image.append(wandb.Image(X[0],caption="Pred:{} Truth:{}".format(label_make(cfg["num_class"],max_indices[0]),y[0])))
    wandb.log({"Exampes":example_image,
                "val_acc": (val_acc / (batch_id+1)*100), 
                "val_loss": val_total_loss / len(val_data)})
    print("-------------->val loss {:.6f} val_acc {:.6f}%".format(val_total_loss / len(val_data),val_acc / (batch_id+1)*100))
    early_stopping((val_acc / (batch_id+1)*100), model2,args,epoch)
    if early_stopping.early_stop:
        print("Early stopping")
        return False
    else:
        return True

def test(device,args, model,model2, dataloader,loss_fn,m,wandb,early_stopping, check_GPU_STATE,cfg):

    if check_GPU_STATE == 2:
        print("GPU USE 2EA")
        print("best_model : ",early_stopping.path2)
        model2.load_state_dict(torch.load(early_stopping.path2))
    else:
        print("GPU USE 1EA")
        print("best_model : ",early_stopping.path)
        model2.load_state_dict(torch.load(early_stopping.path)) 
    test_acc = 0.0
    test_loss = 0.0
    model.eval()
    model2.eval()
    test_total_loss = 0.0

    result = []
    label_result = []
    img_result = []
    y_pred = None
    y_true = None
    con_y_pred = []
    con_y_label = []
    f1_score = 0.0
    save_path = str(args.arch) + "_" + str(cfg["lr"]) + "_" + str(cfg["batch_size"]) + "_" + str(cfg["imgsize"]) + "_" + str(cfg["num_class"]) + "_" + str(cfg["train_loss_pra"]) 
    file_path = os.path.join(args.save_path_report, '{}_report.txt'.format(save_path))
    totla_roc_auc = []
    
    for batch_id, (X, y) in enumerate(dataloader):
        example_image = []
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            model.training = False
            model2.training = False
            # Compute prediction error
            pred = model(X)
            pred2 = model2(X)

            test_loss = loss_fn(pred2, y)
            test_total_loss += test_loss.item()
            _, outputs2 = torch.max(pred2, 1)
            _, outputs3 = torch.max(y, 1)
            con_y_pred.extend(outputs2.cpu().data.numpy())
            con_y_label.extend(outputs3.cpu().data.numpy())
            result.append(outputs2.tolist())
            label_result.append(outputs3.tolist())
        test_acc += calc_accuracy(pred2, y)
    print("-------------->test loss {:.6f} test_acc {:.6f}%".format(test_total_loss / len(dataloader),test_acc / (batch_id+1)*100))
    wandb.log({"test_acc": (test_acc / (batch_id+1)*100), "test_loss": test_total_loss / len(dataloader)})
    confusion(con_y_label,con_y_pred,args.classes,args,save_path)
    report(file_path,con_y_label,con_y_pred,args.classes)
    print("end")
    wandb.finish()

