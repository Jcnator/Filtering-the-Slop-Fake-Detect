import argparse
from ast import arg
import os
import csv
import torch
import torch.nn
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, precision_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import sys
from models import get_model
from PIL import Image 
import pickle
from tqdm import tqdm
from io import BytesIO
from copy import deepcopy
from dataset_reduced_paths import REDUCED_PATHS
import random
import shutil
import joblib
from scipy.ndimage import gaussian_filter
import time

import patcher.cam as cam
import patcher.grad_cam as grad_cam
import cv2

from train import SVM, LinearProbe
from options.util import get_key_from_fake_image_dir

SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}





def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"

    N = y_true.shape[0]

    if y_pred[0:N//2].max() <= y_pred[N//2:N].min(): # perfectly separable case
        return (y_pred[0:N//2].max() + y_pred[N//2:N].min()) / 2 

    best_acc = 0 
    best_thres = 0 
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp>=thres] = 1 
        temp[temp<thres] = 0 

        acc = (temp == y_true).sum() / N  
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc 
    
    return best_thres
        

 
def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality) # ranging from 0-95, 75 is default
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

    return Image.fromarray(img)



def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc    


def calculate_acc_classification(y_true, y_pred):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0])
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1])
    acc = accuracy_score(y_true, y_pred)
    return r_acc, f_acc, acc  


def validate(model, classifier, loader, opt, find_thres=False):

    with torch.no_grad():
        y_true, y_pred = [], []
        count = 0
        print ("Length of dataset: %d" %(len(loader)))
        start = time.process_time()
        for img, label in loader:
            print(100*count/len(loader), '%')
            # print("Label", label.shape)
            count = count + 1
            in_tens = img.cuda()
            feats = model(in_tens)
            pred = classifier(feats)
            pred = classifier.get_validation_y_pred(pred.detach())
            #print(pred)
            y_pred.extend(pred)
            y_true.extend(label.flatten().tolist())
        print("Validation done in", time.process_time() - start, "seconds")


    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Get AP 
    print(len(y_true))
    print(len(y_pred))
    ap = average_precision_score(y_true, y_pred)

    if opt.classifier == "SVM":
        r_acc0, f_acc0, acc0 = calculate_acc_classification(y_true, y_pred)
        return ap, r_acc0, f_acc0, acc0, None, None, None, None

    else:

        # Acc based on 0.5
        r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
        if not find_thres:
            return ap, r_acc0, f_acc0, acc0

        # Acc based on the best thres
        best_thres = find_best_threshold(y_true, y_pred)
        r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)

        return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres


def predict_single_image(model, classifier, image_path, opt):
    stat_from = "imagenet" if opt.arch.lower().startswith("imagenet") else "clip"

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
    ])

    
    with torch.no_grad():
        y_pred = []
        image = transform(Image.open(image_path).convert("RGB"))
        in_tens = image.unsqueeze(0).cuda()
        feats = model(in_tens)
        pred = classifier(feats)
        pred = classifier.get_validation_y_pred(pred.detach())
        #print(pred)
        y_pred.extend(pred)
            
        if opt.classifier == "SVM":
            if y_pred[0] == 0:
                print("Real!")
            else:
                print("Fake!")
        else:
            if y_pred[0] < 0.5:
                print("Real!")
            else:
                print("Fake!")
       


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 




def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg", "bmp"]):
    print(rootdir)
    out = [] 
    for r, d, f in os.walk(rootdir):
        #print(f)
        for file in f:
            if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [ item for item in image_list if must_contain in item   ]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list


class RealFakeDataset(Dataset):
    def __init__(self,  real_path, 
                        fake_path, 
                        data_mode, 
                        max_sample,
                        arch,
                        jpeg_quality=None,
                        gaussian_sigma=None,
                        train_test_split=0.9):

        assert data_mode in ["wang2020"]
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        
        # = = = = = = data path = = = = = = = = = # 
        if type(real_path) == str and type(fake_path) == str:
            real_list, fake_list = self.read_path(real_path, fake_path, data_mode, max_sample)
            val_idx = int(len(real_list)*train_test_split)
            real_list = real_list[val_idx:]
            fake_list = fake_list[val_idx:]
        else:
            real_list = []
            fake_list = []
            for real_p, fake_p in zip(real_path, fake_path):
                real_l, fake_l = self.read_path(real_p, fake_p, data_mode, max_sample)
                val_idx = int(len(real_l)*train_test_split)
                real_list = real_list[val_idx:]
                fake_list = fake_list[val_idx:]
                real_list += real_l
                fake_list += fake_l

        self.total_list = real_list + fake_list


        # = = = = = =  label = = = = = = = = = # 

        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        self.transform = transforms.Compose([
            #transforms.Resize((224,224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
        ])


    def read_path(self, real_path, fake_path, data_mode, max_sample):

        if data_mode == 'wang2020':
            real_list = get_list(real_path, must_contain='0_real')
            fake_list = get_list(fake_path, must_contain='1_fake')


        #print(real_list)
        #print(fake_list)

        if max_sample is not None:
            if (max_sample > len(real_list)) or (max_sample > len(fake_list)):
                max_sample = 100
                print("not enough images, max_sample falling to 100")
            random.shuffle(real_list)
            random.shuffle(fake_list)
            real_list = real_list[0:max_sample]
            fake_list = fake_list[0:max_sample]

        assert len(real_list) == len(fake_list)  

        return real_list, fake_list



    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        
        img_path = self.total_list[idx]

        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")

        if self.gaussian_sigma is not None:
            img = gaussian_blur(img, self.gaussian_sigma) 
        if self.jpeg_quality is not None:
            img = png2jpg(img, self.jpeg_quality)

        img = self.transform(img)
        return img, label





if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--real_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--fake_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--image_path', type=str, default=None, help='path to evaluate a single image')
    parser.add_argument('--max_sample', type=int, default=1000, help='only check this number of images for both fake/real')

    parser.add_argument('--arch', type=str, default='CLIP:ViT-L/14')
    parser.add_argument('--classifier', type=str, default='SVM')

    parser.add_argument('--result_folder', type=str, default='result', help='')
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--jpeg_quality', type=int, default=None, help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None")
    parser.add_argument('--gaussian_sigma', type=int, default=None, help="0,1,2,3,4.     Used to test robustness of our model. Not apply if None")

    parser.add_argument('--train_svm', action='store_true')


    opt = parser.parse_args()

    
    if os.path.exists(opt.result_folder):
        shutil.rmtree(opt.result_folder)
    os.makedirs(opt.result_folder)

    model = get_model(opt.arch)
    #state_dict = torch.load(opt.ckpt, map_location='cpu')
    #model.fc.load_state_dict(state_dict)
    print ("Model loaded..")
    model.eval()
    model.cuda()

    if(opt.classifier == 'SVM'):
        if opt.arch == 'CLIP:ViT-L/14':
            classifier = SVM(1024,2)
            classfier_dict = torch.load('./saved_models/CLIP_SVM.pth', weights_only=True)
        elif opt.arch == 'DINO:vit_b_16':
            classifier = SVM(768,2)
            classfier_dict = torch.load('./saved_models/DINO_SVM_Augmented_2.pth', weights_only=True)

    elif(opt.classifier == 'Linear'):
        model.project_feats = True 
        classifier = LinearProbe(768, num_classes=1)
        if opt.arch == 'CLIP:ViT-L/14':
            classfier_dict = torch.load('./saved_models/CLIP_Linear.pth', weights_only=True)
        elif opt.arch == "DINO:vit_b_16":
            classfier_dict = torch.load('./saved_models/DINO_Linear.pth', weights_only=True)
    
    classifier.load_state_dict(classfier_dict)
    classifier.eval()
    classifier.cuda()
    print("Classfier loaded...")
    print(opt)
    if(opt.image_path is not None):
        predict_single_image(model, classifier, opt.image_path, opt)        
    else:

        if (opt.real_path == None) or (opt.fake_path == None):
            #dataset_paths = DATASET_PATHS
            dataset_paths = REDUCED_PATHS
        else:
            key = get_key_from_fake_image_dir(opt.fake_path)
            dataset_paths = [
                dict(
                    real_path=opt.real_path,
                    fake_path=opt.fake_path,
                    data_mode="wang2020",
                    key=key,
                )
            ]



        for dataset_path in (dataset_paths):
            set_seed()

            dataset = RealFakeDataset(  dataset_path['real_path'], 
                                        dataset_path['fake_path'], 
                                        dataset_path['data_mode'], 
                                        opt.max_sample, 
                                        opt.arch,
                                        jpeg_quality=opt.jpeg_quality, 
                                        gaussian_sigma=opt.gaussian_sigma,
                                        )
            loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
           
            ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = validate(model, classifier, loader, opt, find_thres=True,)
            print("Average Precision:", ap)
            print("acc", acc0)
            print("R_acc 0.5 threhsold:", r_acc0)
            print("F_acc 0.5 threhsold:", f_acc0)
            # print("R_acc Best Threshold:", r_acc1)
            # print("F_acc Best Threshold:", f_acc1)
            print("Best Threhold:", best_thres)


            with open( os.path.join(opt.result_folder,'ap.txt'), 'a') as f:
                
                f.write(dataset_path["key"]+': ' + str(round(ap*100, 2))+'\n' )
                
            with open( os.path.join(opt.result_folder,'acc0.txt'), 'a') as f:
                f.write(dataset_path["key"]+': ' + str(round(r_acc0*100, 2))+'  '+str(round(f_acc0*100, 2))+'  '+str(round(acc0*100, 2))+'\n' )

            if opt.classifier == "Linear":
                with open( os.path.join(opt.result_folder,'acc1.txt'), 'a') as f:
                    f.write(dataset_path["key"]+': ' + str(round(r_acc1*100, 2))+'  '+str(round(f_acc1*100, 2))+'  '+str(round(acc1*100, 2))+ '  ' +str(best_thres) + '\n' )
