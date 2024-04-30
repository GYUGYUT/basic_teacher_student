import os
import numpy as np
import torch
from torchvision import transforms as T
from torch.utils import data
from torch.utils.data import Dataset
from PIL import Image,ImageFilter
from tqdm import tqdm

"""
이미지 스크립트
datapath = PATH
batch = 32
shuffle = TRUE
EX)Img_get_loader(datapath, batch_size, shuffle)
"""
def label_make(class_num,label):
    label = [ 1 if (i == label) else 0 for i in range(class_num) ]
    return torch.Tensor(label)
class Img_Datasets_test(Dataset):
    def __init__(self, tvt, dataset, img_idx, label_idx,size):
        Image_Transform = []
        self.img = []
        Image_Transform.append(T.Resize((size,size)))
        Image_Transform.append(T.ToTensor())
        Image_Transform.append(T.Normalize([0.564,0.439, 0.386], [0.231,0.193, 0.173]))  
        Image_Transform = T.Compose(Image_Transform)      
        for i in tqdm(dataset," %s 이미지 변환 중"%tvt):
            img = i[img_idx]
            self.img.append(Image_Transform(img))
        self.labels = [torch.Tensor(i[label_idx]) for i in dataset]
        
    def __getitem__(self, i):
        return self.img[i],self.labels[i]

    def __len__(self):
        return (len(self.labels))

def Img_get_loader2( datapath, batch_size,size,num_workers, shuffle,r_train=False, r_val=False, r_test=False ):
    train = []
    valid = []
    test = []
    idx = 0
    for classes in tqdm(os.listdir(datapath),"%s"%datapath):
        if(classes == "Normal"):
            idx = label_make(2,0)
        elif(classes == "abNormal"):
            idx = label_make(2,1)

        if(classes != ".DS_Store"):
            path = os.path.join(datapath,str(classes))
            import random
            random.seed(1004)
            shuffle_list = os.listdir(path)
            random.shuffle(shuffle_list)
            if(r_train):
                if(classes != "Normal"):
                    for cls in tqdm(shuffle_list[:2000]," %s  train - [img,label] mapping "% classes):
                        IMAGE = Image.open(path+"/"+ cls).convert('RGB')
                        temp2 = []
                        temp2.append(IMAGE)
                        temp2.append(idx)
                        train.append(temp2)
                else:
                    for cls in tqdm(shuffle_list[:20000]," %s  train - [img,label] mapping "% classes):
                        IMAGE = Image.open(path+"/"+ cls).convert('RGB')
                        temp2 = []
                        temp2.append(IMAGE)
                        temp2.append(idx)
                        train.append(temp2)
            elif(r_val):
                if(classes != "Normal"):
                    for cls in tqdm(shuffle_list[2000:2500]," %s valid - [img,label] mapping "% classes):
                        IMAGE = Image.open(path+"/"+ cls).convert('RGB')
                        temp2 = []
                        temp2.append(IMAGE)
                        temp2.append(idx)
                        valid.append(temp2)
                else:
                    for cls in tqdm(shuffle_list[20000:25000]," %s valid - [img,label] mapping "% classes):
                        IMAGE = Image.open(path+"/"+ cls).convert('RGB')
                        temp2 = []
                        temp2.append(IMAGE)
                        temp2.append(idx)
                        valid.append(temp2)
            elif(r_test):
                if(classes != "Normal"):
                    for cls in tqdm(shuffle_list[2500:3000]," %s test - [img,label] mapping "% classes):
                        IMAGE = Image.open(path+"/"+ cls).convert('RGB')
                        temp2 = []
                        temp2.append(IMAGE)
                        temp2.append(idx)
                        test.append(temp2)             
                else:
                    for cls in tqdm(shuffle_list[25000:30000]," %s test - [img,label] mapping "% classes):
                        IMAGE = Image.open(path+"/"+ cls).convert('RGB')
                        temp2 = []
                        temp2.append(IMAGE)
                        temp2.append(idx)
                        test.append(temp2) 
    if(r_train):
        train_dataset = Img_Datasets_test("traing",train,0,1,size)
        trian_data_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                pin_memory=True)

        return trian_data_loader
    elif(r_val):
        val_dataset = Img_Datasets_test("valid",valid,0,1,size)
        val_data_loader = data.DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                pin_memory=True)

        return val_data_loader
    elif(r_test):
        test_dataset = Img_Datasets_test("test",test,0,1,size)
        test_data_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                pin_memory=True)

        return test_data_loader
