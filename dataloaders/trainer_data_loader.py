import os
from random import random, choice, shuffle
import torch
import cv2
from scipy.ndimage.filters import gaussian_filter
from io import BytesIO
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as TF
import torch.utils.data
import numpy as np
from torch.utils.data import Dataset
from PIL import Image 
#import pickle
#import random

MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}


def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg"]):
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=''):

    image_list = recursively_read(path, must_contain)
    return image_list


class RealFakeDatasetTraining(Dataset):
    def __init__(self,  real_path, 
                        fake_path, 
                        data_mode, 
                        arch,
                        train_test_split=[0.8,0.1,0.1], 
                        train=True,
                        loading_mode="single",
                        models=None,
                        opt = None
                        ):
        self.train = train
        assert np.sum(train_test_split) == 1
        self.train_split = train_test_split[0]
        self.val_split = train_test_split[1]
        self.opt = opt
        assert data_mode in ["wang2020", "ours"]

        if loading_mode == "single":
            # Loading a single model to compare
            real_list = get_list( real_path, must_contain='0_real' )
            fake_list = get_list( fake_path, must_contain='1_fake' )
            assert(len(real_list) == len(fake_list))
            train_len = int(self.train_split *len(real_list))
            val_len = int((self.train_split + self.val_split)* len(real_list))            
            if self.train:
                real_list = real_list[:train_len]
                fake_list = fake_list[:int(train_len)]
            else:
                real_list = real_list[train_len:val_len]
                fake_list = fake_list[train_len:val_len]

        else:
            # Loading all models in imbalanced dataset
            real_paths = {}
            fake_paths = {}
            for model in models:
                #print(model)
                real_path = model['real_path']
                real_paths[real_path] = real_path
                fake_path = model['fake_path']
                fake_paths[fake_path] = fake_path
    
            real_list = []
            for real_path in real_paths:
                real_image_list = get_list( real_path, must_contain='0_real' )
                real_train_len = int(self.train_split *len(real_image_list))
                real_val_len = int((self.val_split + self.train_split) *len(real_image_list))
                if train:
                    real_list = real_list + real_image_list[:real_train_len]
                else:
                    real_list = real_list + real_image_list[real_train_len:real_val_len]

            fake_list = []
            for fake_path in fake_paths:
                fake_image_list = get_list( fake_path, must_contain='1_fake' )
                #print(" Total Fake length", len(fake_image_list))
                fake_train_len = int(self.train_split *len(fake_image_list))
                fake_val_len = int((self.val_split + self.train_split) *len(fake_image_list))
                #print(" Fake training len", fake_train_len)
                #print(" Fake val len",fake_val_len)
                if train:
                    fake_list = fake_list + fake_image_list[:fake_train_len]

                else:
                    fake_list = fake_list + fake_image_list[fake_train_len:fake_val_len]

        print(" Data Loader")
        print(" Real Len:", len(real_list))
        print(" Fake len:", len(fake_list))
        # print(np.array(real_list).shape,np.array(fake_list).shape)

        # = = = = = =  label = = = = = = = = = # 

        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        self.total_list = real_list + fake_list
        shuffle(self.total_list)

        stat_from = "imagenet" if opt.arch.lower().startswith("imagenet") else "clip"
        
        self.transform = transforms.Compose([
            transforms.RandomCrop(opt.cropSize),
            transforms.RandomHorizontalFlip()
        ])

        if train:
            crop_func = transforms.RandomCrop(opt.cropSize)
        else:
            crop_func = transforms.CenterCrop(opt.cropSize)



        self.normalize = transforms.Compose([
            crop_func,
            transforms.ToTensor(),
            transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
        ])       

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")
        img = custom_resize(img, self.opt)
        if self.train:
            img = data_augment(img, self.opt)
            img = self.transform(img)
        img = self.normalize(img)
        return img, label


def data_augment(img, opt):
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}

def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])
