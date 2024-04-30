import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pickle import TRUE


from dataloader2 import *
from cal_accuracy import *
from getmodel  import *

from tqdm import tqdm

from report_save import *
from paser_args import *
from activate_fun import *
from confusion_matrix import *

from parallel import DataParallelModel, DataParallelCriterion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
numworks = 0
#hp
model = None
cfg = { model : "resnet50",
        "epoch" : 100,
        "lr" : 0.000001,
        "batch_size": 256,
        "shuffle" : TRUE,
        "imgsize" : 224,
        "num_class" : 2,
        "numworks" : 0}

args = parse_args("ATMIX_model_50")
model = getModel(args.arch, cfg["num_class"])
best_path = r"/home/ami3/Desktop/GYU/eyes/v5_2class/best_two_gpu/Two_GPU_VersionATMIX_model_50_best_17.pt"
if torch.cuda.device_count() > 1:
  
  best_path = r"/home/ami3/Desktop/GYU/eyes/v5_2class/best_two_gpu/Two_GPU_VersionATMIX_model_50_best_17.pt"
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model,device_ids=[0,1])

model.to(device)

criterion = nn.BCELoss()
m = nn.Sigmoid()
optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

#DATA_PATH
data_path = r"/home/ami3/Desktop/GYU/eyes/dataset/temp"

#여기 바꿔야함
test_data = Img_get_loader2(data_path, cfg["batch_size"],cfg["imgsize"],cfg["numworks"], cfg["shuffle"],r_test=True)
print("best_model",best_path)
model.load_state_dict(torch.load(best_path))

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


test_acc_list=[]
test_loss_list=[]

for batch_id, (img,label) in enumerate(tqdm(test_data," test!!!!")):
    with torch.no_grad():
        model.training = False
        outputs  = model(img.to(device))
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
confusion(con_y_label,con_y_pred,args.classes,args)
report(file_path,con_y_label,con_y_pred)
print("end")