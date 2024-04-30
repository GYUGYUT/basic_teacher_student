from html.entities import name2codepoint
from torchvision import models
import torch.nn as nn
import torch
#from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
from CBAM_GIT import *
from torch import Tensor
from typing import Any, Callable, List, Optional, Type, Union

def getModel(backbone,numclass):
    net = None
    net2 = None
    if backbone == 'resnet50': 
        net = models.resnet50(pretrained=True)
        net.fc = nn.Linear(2048, numclass)
    elif backbone == 'resnet50_gyu_at_student': 
        net = models.resnet50_gyu_at(pretrained=True)
        net.fc = nn.Sequential( nn.Linear(2048, 1024),
                                        nn.Dropout(0.2),
                                        nn.Linear(1024,numclass) )
        net.load_state_dict(torch.load(r'/home/gpu7/eyes/eyes5class/result/One_GPU_Version_resnet50_gyu_at_0.0001_256_224_5_train_50.pt'), strict=False)#학습 가중치 로드 ( 학습 가중치 경로 )

        net2 = models.resnet50_gyu_student(pretrained=True)
        net2.fc = nn.Sequential( nn.Linear(1024,512),
                                                nn.Dropout(0.5),
                                                nn.Linear(512,5) )
        def initialize_weights(model):
            if isinstance(model, nn.Conv2d) or isinstance(model, nn.Linear):
                nn.init.xavier_uniform_(model.weight)
                if model.bias is not None:
                    nn.init.zeros_(model.bias)
        net2.apply(initialize_weights)
    elif backbone == 'mobilenet_teacher': 
        net = models.mobilenet_v3_large(pretrained=True)
        net.classifier.append(nn.Linear(1000, numclass))
        net.load_state_dict(torch.load(r'/home/gpu7/eyes/new_test/dataaug_mobile/best_one_gpu/One_GPU_Version_mobilenet_v3_large_0.0001_32_[224, 224]_5_best_21.pt'), strict=False)#학습 가중치 로드 ( 학습 가중치 경로 )
        net2 = models.mobilenet_v3_small(pretrained=True)
        net2.classifier.append(nn.Linear(1000, numclass))
    elif backbone == 'mobilenet_teacher_hardswish_dropout': 
        net = models.mobilenet_v3_large(pretrained=True)
        net.classifier.append(nn.Linear(1000, numclass))
        net.load_state_dict(torch.load(r'/home/gpu7/eyes/new_test/dataaug_mobile/best_one_gpu/One_GPU_Version_mobilenet_v3_large_0.0001_32_[224, 224]_5_best_21.pt'), strict=False)#학습 가중치 로드 ( 학습 가중치 경로 )
        net2 = models.mobilenet_v3_small(pretrained=True)
        net2.classifier.append(nn.Hardswish())
        net2.classifier.append(nn.Dropout(0.5))
        net2.classifier.append(nn.Linear(1000, numclass))
    elif backbone == 'resnet101_teacher_mobilenet_student': 
        net = models.resnet101(pretrained=True)
        net.fc = nn.Sequential(net.fc,
                                        nn.Linear(1000, numclass))
        net.load_state_dict(torch.load(r'/home/gpu7/eyes/new_test/data_aug/best_one_gpu/One_GPU_Version_resnet101_0.0001_256_[224, 224]_5_best_21.pt'), strict=False)#학습 가중치 로드 ( 학습 가중치 경로 )

        net2 = models.mobilenet_v3_small(pretrained=True)
        net2.classifier.append(nn.Linear(1000, numclass))

    elif backbone == 'densenet121_teacher_mobilenet_student': 
        net = models.densenet121(pretrained=True)
        net.classifier = nn.Sequential(net.classifier,
                                        nn.Linear(1000, numclass))
        net.load_state_dict(torch.load(r'/home/gpu7/best_one_gpu/One_GPU_Version_densenet121_0.0001_32_[512, 512]_5_best_18.pt'), strict=False)#학습 가중치 로드 ( 학습 가중치 경로 )
        net2 = models.mobilenet_v3_small(pretrained=True)
        net2.classifier.append(nn.Linear(1000, numclass))

    elif backbone == 'densenet121_teacher_mobilenet_student_dropout': 
        net = models.densenet121(pretrained=True)
        net.classifier = nn.Sequential(net.classifier,
                                        nn.Linear(1000, numclass))
        net.load_state_dict(torch.load(r'/home/gpu7/best_one_gpu/One_GPU_Version_densenet121_0.0001_32_[512, 512]_5_best_18.pt'), strict=False)#학습 가중치 로드 ( 학습 가중치 경로 )
        net2 = models.mobilenet_v3_small(pretrained=True)
        net2.classifier.append(nn.Hardswish())
        net2.classifier.append(nn.Dropout(0.5))
        net2.classifier.append(nn.Linear(1000, numclass))

    elif backbone == 'densenet121_teacher_mobilenet_student_dropout_03': 
        net = models.densenet121(pretrained=True)
        net.classifier = nn.Sequential(net.classifier,
                                        nn.Linear(1000, numclass))
        net.load_state_dict(torch.load(r'/home/gpu7/best_one_gpu/One_GPU_Version_densenet121_0.0001_32_[512, 512]_5_best_18.pt'), strict=False)#학습 가중치 로드 ( 학습 가중치 경로 )
        net2 = models.mobilenet_v3_small(pretrained=True)
        net2.classifier.append(nn.Hardswish())
        net2.classifier.append(nn.Dropout(0.3))
        net2.classifier.append(nn.Linear(1000, numclass))
    return net,net2