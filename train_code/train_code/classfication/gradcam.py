import torchvision.models as models
import torch.nn as nn
import torch
import os
import cv2
import PIL
import torchvision
import torchvision.transforms as transforms
import datetime
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from torchvision.utils import make_grid, save_image
from torch.autograd import Function
import shutil
from torchvision.models import resnet50, mobilenet_v2,resnet101
from getmodel import *
def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

img_size = 224

def model_c():
    
    print('Finished Train_val')
    best_path = r"/home/ami-3/Desktop/ohgyutae/eyes/image_train/best_one_gpu1/One_GPU_Version_resnet50_best_21.pt"
    print("best_model",best_path)
    device = torch.device('cpu')
    model = getModel("resnet50", 2)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    model.cuda()
    return model

def Normal():
    img_path = r"/home/ami-3/Desktop/ohgyutae/eyes/dataset_test3/NORMAL_574.jpg"
    img_path2 = r"/home/ami-3/Desktop/ohgyutae/eyes/dataset_test3/NORMAL_732.jpg"
    img_path3 = r"/home/ami-3/Desktop/ohgyutae/eyes/dataset_test3/NORMAL_178.jpg"
    label_check = 0
    pathlist = []
    pathlist.append(img_path)
    pathlist.append(img_path2)
    pathlist.append(img_path3)
    return pathlist

def abNormal():
    img_path = r"/home/ami-3/Desktop/ohgyutae/eyes/dataset_test3/UCDLOW_20.jpg"
    img_path2 = r"/home/ami-3/Desktop/ohgyutae/eyes/dataset_test3/PK_36.jpg"
    img_path3 = r"/home/ami-3/Desktop/ohgyutae/eyes/dataset_test3/conjunctivitis_15.jpg"
    label_check = 1
    pathlist = []
    pathlist.append(img_path)
    pathlist.append(img_path2)
    pathlist.append(img_path3)
    return pathlist



def imgsave(img_path,label_check,saved_loc,model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("img_path debugging :", os.path.isfile(img_path))
    print("img_path :", img_path)
    pil_img = PIL.Image.open(img_path).convert('RGB').resize((img_size, img_size), PIL.Image.LANCZOS)
    def normalize(tensor, mean, std):
        if not tensor.ndimension() == 4:
            raise TypeError('tensor should be 4D')

        mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
        std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

        return tensor.sub(mean).div(std)


    class Normalize(object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            return self.do(tensor)
        
        def do(self, tensor):
            return normalize(tensor, self.mean, self.std)
        def __repr__(self):
            return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

    normalizer = Normalize(mean=[0.56450975,0.439657, 0.38688567], std=[0.23105052,0.19330876, 0.17322437])
    torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
    torch_img = F.interpolate(torch_img, size=(img_size, img_size), mode='bilinear', align_corners=False) # (1, 3, 224, 224)
    normed_torch_img = normalizer(torch_img) # (1, 3, 224, 224)



    
    if os.path.exists(saved_loc):
        shutil.rmtree(saved_loc)
    os.mkdir(saved_loc)

    print("결과 저장 위치: ", saved_loc)

    class GuidedBackpropReLU(Function):
        @staticmethod
        def forward(self, input_img):
            # input image 기준으로 양수인 부분만 1로 만드는 positive_mask 생성
            positive_mask = (input_img > 0).type_as(input_img)
            
            # torch.addcmul(input, tensor1, tensor2) => output = input + tensor1 x tensor 2
            # input image와 동일한 사이즈의 torch.zeros를 만든 뒤, input image와 positive_mask를 곱해서 output 생성
            output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
            
            # backward에서 사용될 forward의 input이나 output을 저장
            self.save_for_backward(input_img, output)
            return output

        @staticmethod
        def backward(self, grad_output):
            
            # forward에서 저장된 saved tensor를 불러오기
            input_img, output = self.saved_tensors
            grad_input = None

            # input image 기준으로 양수인 부분만 1로 만드는 positive_mask 생성
            positive_mask_1 = (input_img > 0).type_as(grad_output)
            
            # 모델의 결과가 양수인 부분만 1로 만드는 positive_mask 생성
            positive_mask_2 = (grad_output > 0).type_as(grad_output)
            
            # 먼저 모델의 결과와 positive_mask_1과 곱해주고,
            # 다음으로는 positive_mask_2와 곱해줘서 
            # 모델의 결과가 양수이면서 input image가 양수인 부분만 남도록 만들어줌
            grad_input = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img),
                                    torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), grad_output,
                                                    positive_mask_1), positive_mask_2)
            return grad_input

    class GuidedBackpropReLUModel:
        def __init__(self, model, use_cuda):
            self.model = model
            self.model.eval()
            self.cuda = use_cuda
            if self.cuda:
                self.model = model.cuda()

            def recursive_relu_apply(module_top):
                for idx, module in module_top._modules.items():
                    recursive_relu_apply(module)
                    if module.__class__.__name__ == 'ReLU':
                        module_top._modules[idx] = GuidedBackpropReLU.apply

            # replace ReLU with GuidedBackpropReLU
            recursive_relu_apply(self.model)

        def forward(self, input_img):
            return self.model(input_img)

        def __call__(self, input_img, target_category=None):
            if self.cuda:
                input_img = input_img.cuda()

            input_img = input_img.requires_grad_(True)

            output = self.forward(input_img)

            if target_category is None:
                target_category = np.argmax(output.cpu().data.numpy())

            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0][target_category] = 1
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            if self.cuda:
                one_hot = one_hot.cuda()

            one_hot = torch.sum(one_hot * output)
            # 모델이 예측한 결과값을 기준으로 backward 진행
            one_hot.backward(retain_graph=True)

            # input image의 gradient를 저장
            output = input_img.grad.cpu().data.numpy()
            output = output[0, :, :, :]
            output = output.transpose((1, 2, 0))
            return output
    def deprocess_image(img):
        """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
        img = img - np.mean(img)
        img = img / (np.std(img) + 1e-5)
        img = img * 0.1
        img = img + 0.5
        img = np.clip(img, 0, 1)
        return np.uint8(img * 255)

    # final conv layer name 

    # activations
    feature_blobs = []

    # gradients
    backward_feature = []

    # output으로 나오는 feature를 feature_blobs에 append하도록
    def hook_feature(module, input, output):
        feature_blobs.append(output.cpu().data)
        

    # Grad-CAM
    def backward_hook(module, input, output):
        backward_feature.append(output[0])

    # model.features[-1].register_forward_hook(hook_feature)
    # model.features[-1].register_full_backward_hook(backward_hook)
    model.layer4[-1].register_forward_hook(hook_feature)
    model.layer4[-1].register_full_backward_hook(backward_hook)

    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().detach().numpy()) # [1000, 512]

    # Prediction
    logit = model(normed_torch_img.float())

    # ============================= #
    # ==== Grad-CAM main lines ==== #
    # ============================= #


    # Tabby Cat: 281, pug-dog: 254
    score = logit[:, label_check].squeeze() # 예측값 y^c
    score.backward(retain_graph = True) # 예측값 y^c에 대해서 backprop 진행

    activations = feature_blobs[0].to(device) # (1, 512, 7, 7), forward activations
    gradients = backward_feature[0] # (1, 512, 7, 7), backward gradients
    b, k, u, v = gradients.size()

    alpha = gradients.view(b, k, -1).mean(2) # (1, 512, 7*7) => (1, 512), feature map k의 'importance'
    weights = alpha.view(b, k, 1, 1) # (1, 512, 1, 1)

    grad_cam_map = (weights*activations).sum(1, keepdim = True) # alpha * A^k = (1, 512, 7, 7) => (1, 1, 7, 7)
    grad_cam_map = F.relu(grad_cam_map) # Apply R e L U
    grad_cam_map = F.interpolate(grad_cam_map, size=(224, 224), mode='bilinear', align_corners=False) # (1, 1, 224, 224)
    map_min, map_max = grad_cam_map.min(), grad_cam_map.max()
    grad_cam_map = (grad_cam_map - map_min).div(map_max - map_min).data # (1, 1, 224, 224), min-max scaling

    # grad_cam_map.squeeze() : (224, 224)
    grad_heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_map.squeeze().cpu()), cv2.COLORMAP_JET) # (224, 224, 3), numpy

    # Grad-CAM heatmap save
    cv2.imwrite(os.path.join(saved_loc, "Grad_CAM_heatmap.jpg"), grad_heatmap)

    grad_heatmap = np.float32(grad_heatmap) / 255

    grad_result = grad_heatmap + pil_img
    grad_result = grad_result / np.max(grad_result)
    grad_result = np.uint8(255 * grad_result)

    # Grad-CAM Result save
    cv2.imwrite(os.path.join(saved_loc, "Grad_Result.jpg"), grad_result)

    # ============================= #
    # ==Guided-Backprop main lines= #
    # ============================= #

    # gb_model => ReLU function in resnet50 change to GuidedBackpropReLU.
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
    gb_num = gb_model(torch_img, target_category = 0) # 여기 바꿔야함
    gb = deprocess_image(gb_num) # (224, 224, 3), numpy

    # Guided Backprop save
    cv2.imwrite(os.path.join(saved_loc, "Guided_Backprop.jpg"), gb)

    # Guided-Backpropagation * Grad-CAM => Guided Grad-CAM 
    # See Fig. 2 in paper.
    # grad_cam_map : (1, 1, 224, 224) , torch.Tensor
    grayscale_cam = grad_cam_map.squeeze(0).cpu().numpy() # (1, 224, 224), numpy
    grayscale_cam = grayscale_cam[0, :] # (224, 224)
    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam]) # (224, 224, 3)

    cam_gb = deprocess_image(cam_mask * gb_num)

    # Guided Grad-CAM save
    cv2.imwrite(os.path.join(saved_loc, "Guided_Grad_CAM.jpg"), cam_gb)

    img = cv2.imread(img_path)
    print(img.shape)
    Original_image = cv2.resize(img, dsize=((img_size, img_size)))

    G_heatmap = cv2.cvtColor(cv2.imread(os.path.join(saved_loc, "Grad_CAM_heatmap.jpg")), cv2.COLOR_BGR2RGB)
    G_result = cv2.cvtColor(cv2.imread(os.path.join(saved_loc, "Grad_Result.jpg")), cv2.COLOR_BGR2RGB)
    G_Back = cv2.cvtColor(cv2.imread(os.path.join(saved_loc, "Guided_Backprop.jpg")), cv2.COLOR_BGR2RGB)
    G_CAM = cv2.cvtColor(cv2.imread(os.path.join(saved_loc, "Guided_Grad_CAM.jpg")), cv2.COLOR_BGR2RGB)

    cv2.imwrite(os.path.join(saved_loc, "back_and.jpg"), cv2.bitwise_and(Original_image,G_Back))
    cv2.imwrite(os.path.join(saved_loc, "Gcam_and.jpg"), cv2.bitwise_and(Original_image,G_CAM))
    cv2.imwrite(os.path.join(saved_loc, "back_xor.jpg"), cv2.bitwise_xor(Original_image,G_Back))
    cv2.imwrite(os.path.join(saved_loc, "Gran_xor.jpg"), cv2.bitwise_xor(Original_image,G_CAM))

    back_and = cv2.cvtColor(cv2.imread(os.path.join(saved_loc, "back_and.jpg")), cv2.COLOR_BGR2RGB)
    Gcam_and = cv2.cvtColor(cv2.imread(os.path.join(saved_loc, "Gcam_and.jpg")), cv2.COLOR_BGR2RGB)
    back_xor = cv2.cvtColor(cv2.imread(os.path.join(saved_loc, "back_xor.jpg")), cv2.COLOR_BGR2RGB)
    Gran_xor = cv2.cvtColor(cv2.imread(os.path.join(saved_loc, "Gran_xor.jpg")), cv2.COLOR_BGR2RGB)

    Original_image = cv2.cvtColor(Original_image, cv2.COLOR_BGR2RGB)

    Total = cv2.hconcat([Original_image, G_heatmap, G_result, G_Back, G_CAM])
    Total2 = cv2.hconcat([Original_image, back_and, Gcam_and, back_xor,Gran_xor])
    cv2.imwrite(os.path.join(saved_loc, "Final_result2.jpg"), Total2)

    plt.rcParams["figure.figsize"] = (20, 4)
    plt.imshow(Total)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.savefig(os.path.join(saved_loc, "Final_result.jpg"))
    #plt.show()

def main():
    
    for i in range(2):
        if(i == 0):
            paths = Normal()
            for j in range(len(paths)):
                model = model_c()
                saved_loc = os.path.join('../', "gradcam_resnet50/Normal/")
                saved_loc = os.path.join(saved_loc, str(j)+"/")
                createDirectory(saved_loc)
                imgsave(paths[j],i,saved_loc,model)
        elif(i == 1):
            paths = abNormal()
            for j in range(len(paths)):
                model = model_c()
                saved_loc = os.path.join('../', "gradcam_resnet50/abNormal/")
                saved_loc = os.path.join(saved_loc, str(j)+"/")
                createDirectory(saved_loc)
                imgsave(paths[j],i,saved_loc,model)
main()