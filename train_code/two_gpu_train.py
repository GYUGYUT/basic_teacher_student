import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pickle import TRUE

from dataloader_dir2 import *
from cal_accuracy import *
from getmodel  import *

from plot_show import *
from tqdm import tqdm

from img_save import *
from report_save import *
from paser_args import *
from activate_fun import *
from confusion_matrix import *

# import EarlyStopping
from pytorchtools import EarlyStopping
#DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
numworks = 0

#hp
epoch = 100
lr = 0.000001
batch_size= 128
shuffle = TRUE
imgsize = 224
num_class = 2
check_GPU_STATE = 0
#dataloader
train_data_path = r"../dataset_train3"
val_data_path = r"../dataset_val3"
test_data_path = r"../dataset_test3"

train_data = get_loader(train_data_path,imgsize, batch_size,shuffle)
val_data = get_loader(val_data_path,imgsize, batch_size,shuffle)
print("데이터 변환 성공")

#모델
args = parse_args("ATMIX_model_50_elu")
model = getModel(args.arch, num_class)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    check_GPU_STATE  = 2
    model = nn.DataParallel(model,device_ids=[0,1])
else:
    check_GPU_STATE  = 1
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
model.to(device)

#criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=lr)

print("선정된 모델",args.arch)

#LIST
epochs = []

train_acc_list = []
train_loss_list = []

val_acc_list=[]
val_loss_list=[]

test_acc_list=[]
test_loss_list=[]


print("학습 시작")
best_model = None
best_model_acc = 0
best_model_loss = 1000
temp_idx = 0
best_path = None
best_path2 = None
m = nn.Sigmoid()
early_stopping = EarlyStopping()
for epoch in tqdm(range(1,epoch+1),"전체 진행률"):
    train_loss = 0.0
    train_acc = 0.0
    model.train()
    epochs.append(epoch)
    for batch_id, (img,label) in enumerate(tqdm(train_data," %d  Train!!"% epoch)):
        outputs = model(img.to(device))
        #label = torch.eye(num_class)[label]
        
        train_loss = criterion(m(outputs.to(device)), label.to(device))
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_acc += calc_accuracy(outputs.to(device), label.to(device))

    train_acc_list.append( (train_acc.data.cpu().numpy() / (batch_id+1))*100)
    train_loss_list.append(train_loss.data.cpu().numpy())
    print("epoch {} batch id {} loss {:.6f} train acc {:.6f}%".format(epoch, batch_id+1, (train_loss.data.cpu().numpy()), (train_acc.data.cpu().numpy() / (batch_id+1))*100))
    if(epoch%10 == 0): #pt저장
        path = os.path.join(args.save_path, 'One_GPU_Version_{}_train_{}.pt'.format(args.arch,epoch))
        path2 = os.path.join(args.save_path2, 'Two_GPU_Version_{}_train_{}.pt'.format(args.arch,epoch))
        torch.save(model.state_dict(),path2)
        torch.save(model.module.state_dict(), path)
    val_acc = 0.0
    val_loss = 0.0
    model.eval()
    
    for batch_id, (img,label) in enumerate(tqdm(val_data," %d val!!!!"% epoch)):
        with torch.no_grad():
            model.training = False
            outputs  = model(img.to(device))
            #label = torch.eye(num_class)[label]
            val_loss = criterion(m(outputs.to(device)),label.to(device))
        val_acc += calc_accuracy(outputs.to(device), label.to(device))
    
    val_acc_list.append((val_acc.data.cpu().numpy() / (batch_id+1)*100))
    val_loss_list.append(val_loss.data.cpu().numpy())
    print("-------------->val loss {:.6f} val_acc {:.6f}%".format((val_loss.data.cpu().numpy()),(val_acc.data.cpu().numpy() / (batch_id+1)*100)))
    early_stopping(val_loss_list[-1], model,args,epoch)
    if early_stopping.early_stop:
        print("Early stopping")
        break
print('Finished Train_val')

del train_data 
del val_data

print("best_model",best_path)
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
test_data = get_loader(test_data_path,imgsize, batch_size,shuffle)

for batch_id, (img,label) in enumerate(tqdm(test_data," %d test!!!!"% epoch)):
    with torch.no_grad():
        model.training = False
        outputs  = model(img.to(device))
        #label = torch.eye(num_class)[label]
        test_loss = criterion(m(outputs.to(device)), label.to(device))
        test_acc += calc_accuracy(outputs.to(device), label.to(device))
        _, outputs2 = torch.max(outputs, 1)
        _, outputs3 = torch.max(label, 1)
        con_y_pred.extend(outputs2.cpu().data.numpy())
        con_y_label.extend(outputs3.cpu().data.numpy())
        result.append(outputs2.tolist())
        label_result.append(outputs3.tolist())
    test_acc_list.append((test_acc.data.cpu().numpy() / (batch_id+1)*100))
    test_loss_list.append(test_loss.data.cpu().numpy())
print("-------------->test loss {:.6f} test_acc {:.6f}%".format((test_loss),(test_acc / (batch_id+1)*100)))
show(epochs,train_acc_list,train_loss_list,val_acc_list,val_loss_list,args)
confusion(con_y_label,con_y_pred,args.classes,args)
report(file_path,con_y_label,con_y_pred)
#img_save(args,img_result,label_result,result,imgsize)
print("end")