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
def train(device,args,model,train_data,val_data,loss_fn, optimizer, epoch,m,wandb,early_stopping):
    model.train()
    loss = 0.0
    total_loss = 0
    train_acc = 0.0
    for batch_id, (X, y) in enumerate(tqdm(train_data," %d  Train!!"% epoch)):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(m(pred), y)
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc += calc_accuracy(pred, y)
    wandb.log({"train_acc": (train_acc / (batch_id+1))*100, "train_loss": total_loss / len(train_data)},step=epoch)    
    print("epoch {} batch id {} loss {:.6f} train acc {:.6f}%".format(epoch, batch_id+1, (total_loss / len(train_data)), (train_acc/ (batch_id+1))*100))
    if(epoch%10 == 0): #pt저장
        path = os.path.join(args.save_path, 'One_GPU_Version_{}_train_{}.pt'.format(args.arch,epoch))
        path2 = os.path.join(args.save_path2, 'Two_GPU_Version_{}_train_{}.pt'.format(args.arch,epoch))
        torch.save(model.state_dict(),path2)
        torch.save(model.module.state_dict(), path)
    
    val_acc = 0.0
    val_loss = 0.0
    model.eval()
    val_total_loss = 0.0
    for batch_id, (X, y) in enumerate(tqdm(val_data," %d val!!!!"% epoch)):
        example_image = []
        X, y = X.to(device), y.to(device)

        with torch.no_grad():
            model.training = False
            # Compute prediction error
            pred = model(X)
            val_loss = loss_fn(m(pred), y)
            val_total_loss += val_loss.item()

        val_acc += calc_accuracy(pred, y)
        example_image.append(wandb.Image(X[0],caption="Pred:{} Truth:{}".format(pred[0],y[0])))
    wandb.log({"Exampes":example_image,
                "val_acc": (val_acc / (batch_id+1)*100), 
                "val_loss": val_total_loss / len(val_data)})
    print("-------------->val loss {:.6f} val_acc {:.6f}%".format(val_total_loss / len(val_data),val_acc / (batch_id+1)*100))
    early_stopping(val_total_loss / len(val_data), model,args,epoch)
    if early_stopping.early_stop:
        print("Early stopping")
        return False
    else:
        return True

def test(device,args, model, dataloader,loss_fn,m,wandb,early_stopping, check_GPU_STATE):

    if check_GPU_STATE == 2:
        print("GPU USE 2EA")
        print("best_model : ",early_stopping.path2)
        model.load_state_dict(torch.load(early_stopping.path2))
    else:
        print("GPU USE 1EA")
        print("best_model : ",early_stopping.path)
        model.load_state_dict(torch.load(early_stopping.path)) 
    test_acc = 0.0
    test_loss = 0.0
    model.eval()
    test_total_loss = 0.0

    result = []
    label_result = []
    img_result = []
    y_pred = None
    y_true = None
    con_y_pred = []
    con_y_label = []
    f1_score = 0.0
    file_path = os.path.join(args.save_path_report, '{}_report.txt'.format(args.arch))
    totla_roc_auc = []
    
    for batch_id, (X, y) in enumerate(dataloader):
        example_image = []
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            model.training = False
            # Compute prediction error
            pred = model(X)

            test_loss = loss_fn(m(pred.to(device)), y.to(device))
            test_total_loss += test_loss.item()
            _, outputs2 = torch.max(pred, 1)
            _, outputs3 = torch.max(y, 1)
            con_y_pred.extend(outputs2.cpu().data.numpy())
            con_y_label.extend(outputs3.cpu().data.numpy())
            result.append(outputs2.tolist())
            label_result.append(outputs3.tolist())
        test_acc += calc_accuracy(pred, y)
    print("-------------->test loss {:.6f} val_acc {:.6f}%".format(test_total_loss / len(dataloader),test_acc / (batch_id+1)*100))
    wandb.log({"test_acc": (test_acc / (batch_id+1)*100), "test_loss": test_total_loss / len(dataloader)})
    confusion(con_y_label,con_y_pred,args.classes,args)
    report(file_path,con_y_label,con_y_pred)
    print("end")
    wandb.finish()

