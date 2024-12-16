import cv2
import os
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from PIL import Image
from PIL import ImageFile

def mask_weights(weights, indices, cls):
    pass



def reshape_transform(tensor, height = 16, width = 16):
    b, p, w = tensor.shape
    reshaped = torch.reshape(tensor[:,1:,:],(b,height, width, w))
    reshaped = reshaped.transpose(0,2,1)
    return reshaped

def get_weights(model, i=1):
    # print("weights")
    #print(model.model.visual.transformer.resblocks.23.ln_2)
    #for weight in list(model.model.visual.transformer.parameters()):
        #print(weight.shape)
    # print(list(model.model.visual.transformer.parameters()))
    return list(model.model.visual.transformer.parameters())[-i]

def generate_CAM(model, features, image, output_name, result_path):
    # inputs
    # feature
    # weight
    # class_idx
    # output
    # print("CAM")
    weights  = get_weights(model)

    c, h, w = image.shape
    image_size = (224,224)
    features = features[:,1:,:]
    features = torch.reshape(features,(16,16,1024))
    # print(" weights", weights.shape)
    # print(" features", features.shape)
    #print(features)
    #print(weights)
    cam = torch.matmul(features, weights)
    # print(" Cam vals", cam.min(), cam.max())
    # print(" Cam average", torch.mean(cam))
    softmax = torch.nn.Softmax()
    #cam = torch.logit(cam)
    sig = torch.nn.Sigmoid()
    # cam = softmax(cam)
    cam = sig(cam)
    
    # print(" Post Cam vals", cam.min(), cam.max())
    # print(" Post Cam average", torch.mean(cam))
    # print(" cam shape", cam.shape)

    # print(" vals", cam.min(), cam.max())

    # cam = cam - torch.min(cam)
    # cam_img = cam / torch.max(cam)
    cam_img = (cam * 255).to(torch.uint8)
    cam_img = cam_img.to("cpu").numpy() 
    cam_img = cv2.resize(cam_img, image_size)
    heat_map = cv2.applyColorMap(cv2.resize(cam_img,(h, w)), cv2.COLORMAP_JET)
    

    # print("image", image.shape)
    image = torch.permute(image, (1,2,0))
    # print("permuted image", image.shape)
    image = image.to('cpu').numpy()
    image = image - image.min()
    image = image / image.max()
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # cv2.imwrite(os.path.join(result_path, f"{output_name}i.png"),image)
    # cv2.imwrite(os.path.join(result_path, f"{output_name}h.png"),heat_map)
    # print("image", image.min(), image.max())
    # print("heat map", cam_img.min(), cam_img.max())

    display_image = (image*0.5 + heat_map*0.5)
    # print(display_image)
    # print("display_image", display_image.min(), display_image.max())
    cv2.imwrite(os.path.join(result_path, f"{output_name}r.png"),display_image)


        
def generate_cam_alt(model, features, image):
    c, h, w = image.shape
    weights = get_weights(model,)
    image_size = (224,224)

    features = features[:, 0, :]
    features = torch.reshape(features,(32,32))
    