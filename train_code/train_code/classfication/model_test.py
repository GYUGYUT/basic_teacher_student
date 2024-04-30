
from torchvision import models
from report_save import *
from paser_args import *
from activate_fun import *
from confusion_matrix import *
from torchvision import models
import torchsummary

from torchvision import models
import torch.nn as nn
import torch
from getmodel import *

model = models.regnet_y_32gf()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

torchsummary.summary(model, (3, 224,224),device='cuda')
