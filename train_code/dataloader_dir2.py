import os
from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import numpy as np
import random
import torch

def label_make(class_num,label):
    label = [ 1 if (i == label) else 0 for i in range(class_num) ]
    return torch.Tensor(label)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 시드 설정
seed = 42
set_seed(seed)
class AddGaussianNoise():
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class ImageFolder(data.Dataset):
    def __init__(self, Image_path,img_size):
        self.Image_path = Image_path
        self.filelist = os.listdir(Image_path)
        self.img_size = img_size

    def __getitem__(self, index):
        filename = self.filelist[index]
        IMAGE = Image.open(os.path.join(self.Image_path , filename)).convert('RGB')

        Image_Transform = []
        Image_Transform.append(T.Resize((self.img_size[0], self.img_size[1])))
        Image_Transform.append(T.ToTensor())
        Image_Transform.append(T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        select_num = random.randint(0,4)
        if select_num == 0:
            pass
        elif select_num == 1:
            Image_Transform.append(T.RandomHorizontalFlip())
        elif select_num == 2:
            Image_Transform.append(AddGaussianNoise(0, 1))
        elif select_num == 3:
            Image_Transform.append(T.RandomRotation([-8, 8]))
        elif select_num == 4:
            Image_Transform.append(T.RandomVerticalFlip())

        Image_Transform = T.Compose(Image_Transform)
        
        IMAGE = Image_Transform(IMAGE)
        label = None
        if(filename.split('_')[0] == "Cataract"):
            label = label_make(5,0)
        elif(filename.split('_')[0] == "Corneal"):
            label = label_make(5,1)
        elif(filename.split('_')[0] == "Eyelid"):
            label = label_make(5,2)
        elif(filename.split('_')[0] == "Normal"):
            label = label_make(5,3)
        elif(filename.split('_')[0] == "Uveitis"):
            label = label_make(5,4)
        else:
            print(filename.split('_')[0])
            print("stop")

        return IMAGE, label
    def __len__(self):
        """Returns the total number of font files."""
        return len(self.filelist)
class ImageFolder2(data.Dataset):
    def __init__(self, Image_path,img_size):
        self.Image_path = Image_path
        self.filelist = os.listdir(Image_path)
        self.img_size = img_size

    def __getitem__(self, index):
        filename = self.filelist[index]
        IMAGE = Image.open(os.path.join(self.Image_path , filename)).convert('RGB')

        Image_Transform = []
        Image_Transform.append(T.Resize((self.img_size[0], self.img_size[1])))
        Image_Transform.append(T.ToTensor())
        Image_Transform.append(T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        Image_Transform = T.Compose(Image_Transform)
        
        IMAGE = Image_Transform(IMAGE)
        label = None
        if(filename.split('_')[0] == "Cataract"):
            label = label_make(5,0)
        elif(filename.split('_')[0] == "Corneal"):
            label = label_make(5,1)
        elif(filename.split('_')[0] == "Eyelid"):
            label = label_make(5,2)
        elif(filename.split('_')[0] == "Normal"):
            label = label_make(5,3)
        elif(filename.split('_')[0] == "Uveitis"):
            label = label_make(5,4)
        else:
            print(filename.split('_')[0])
            print("stop")

        return IMAGE, label

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.filelist)


def get_loader(Image_path,img_size, batch_size, shuffle):
    """Builds and returns Dataloader."""
    dataset = ImageFolder(Image_path,img_size)
    data_loader = data.DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            pin_memory = True,
                            num_workers=8)
    return data_loader

def get_loader2(Image_path,img_size, batch_size, shuffle):
    """Builds and returns Dataloader."""
    dataset = ImageFolder2(Image_path,img_size)
    data_loader = data.DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            pin_memory = True,
                            num_workers=8)
    return data_loader