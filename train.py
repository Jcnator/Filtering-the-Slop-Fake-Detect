import argparse
from ast import arg
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

import torchvision.transforms.v2 as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from torch.utils.data import Dataset
import sys
from models import get_model
from PIL import Image 
#import pickle
from tqdm import tqdm
from io import BytesIO
from copy import deepcopy
from dataset_reduced_paths import REDUCED_PATHS
import random
import shutil
import time
from dataloaders.trainer_data_loader import RealFakeDatasetTraining
#from validate import calculate_acc

class SVM(nn.Module):
    # https://github.com/nathanlem1/SVM_PyTorch/blob/master/SVM_PyTorch_train.py
    """
    Using fully connected neural network to implement linear SVM and Logistic regression with hinge loss and
    cross-entropy loss which computes softmax internally, respectively.
    """
    def __init__(self, input_size=1024, num_classes=2, class_ratio=5):
        super(SVM, self).__init__()    # Call the init function of nn.Module
        self.fc = nn.Linear(input_size, num_classes)
        self.loss_fn = nn.MultiMarginLoss(weight=torch.tensor([1.0,1/class_ratio]).cuda())

    def forward(self, x):
        out = self.fc(x)
        return out
    
    def get_loss(self, pred, label):
        loss = self.loss_fn(pred, label.cuda())
        return loss
    
    def get_accuracy(self, pred, label):
        values, indices = torch.max(pred, dim=1)
        accuracy = accuracy_score(label, indices.detach().cpu().numpy())
        return accuracy
    
    def get_validation_y_pred(self, pred):
        values, indices = torch.max(pred, dim=1)
        return indices.cpu()


# Original Linear Probing Model for baseline comparison
class LinearProbe(nn.Module):
    def __init__(self, input_size=768, num_classes=1, class_ratio=5):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.loss_fn = nn.BCEWithLogitsLoss(weight=torch.tensor([1/class_ratio]).cuda())

    def forward(self, x):
        out = self.fc(x)
        return out
    
    def get_loss(self, pred, label):
        #print("pred", pred.shape)
        #print(pred)
        #print("label", label.shape)
        pred = pred.squeeze(1)
        loss = self.loss_fn(pred, label.float().cuda())
        return loss
    
    def get_accuracy(self, pred, label):
        thres = 0.5
        #values, indices = torch.max(pred, dim=1)
        acc = accuracy_score(label, pred.sigmoid().flatten().detach().cpu().numpy() > thres)
      
        return acc
    
    def get_validation_y_pred(self, pred):
        return pred.sigmoid().flatten().cpu()


def train(classfier, encoder, loaders, optimizer, scheduler, logging, weight_path, num_epochs=50):
    
    # Early stopping
    best_loss = float('inf')
    early_stop_patience = 10

    train_loader = loaders["train_loader"]
    
    print("Train Loader Len:", len(train_loader))
    start_time = time.time()


    for e in range(num_epochs):
        train_loss = 0
        curr_batch = 0
        for image, label in train_loader:
            batch_start_time = time.time()

            in_tens = image.cuda()
            feats = encoder(in_tens)
            #print(" Encoded Feats", feats.shape, feats.device)
            pred = classfier(feats)
        
            loss = classfier.get_loss(pred, label)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_time = time.time()
            curr_batch += 1
            if curr_batch % 25 == 0:
                print(" Batch %d out of %d. Batch loss: %f done in %f seconds" %(curr_batch, len(train_loader), loss, batch_time - batch_start_time ))
            
        train_loss = train_loss / len(train_loader)
        print("[%d] Train loss: %f" %(e, train_loss))

        # validation
        test_loss = 0
        val_loader = loaders["val_loader"]
        accuracy = 0
        with torch.no_grad():
            for image, label in val_loader:
                in_tens = image.cuda()
                feats = encoder(in_tens)
                pred = classfier(feats) 
                test_loss += classfier.get_loss(pred, label)
                accuracy += classfier.get_accuracy(pred, label)

        test_loss = test_loss / len(val_loader)
        epoch_time = time.time()
        print("[%d] Val loss: %f, Val Accuracy: %f done in %f" %(e, test_loss, accuracy /len(val_loader), epoch_time - start_time))

        if logging:
            wandb.log({"loss": test_loss, 
                       "training_loss": train_loss,
                       "accuracy": accuracy /len(val_loader)
                       })

        if(test_loss < best_loss):
            best_loss = test_loss
            best_network = classfier.state_dict()
        else:
            early_stop_patience -= 1
            print("Patience Running Out...", early_stop_patience)
            if early_stop_patience == 0:
                print("Patience Ran Out. Early Stopping")
                print('saving net...')
                torch.save(best_network, 'best_network.pth')
                break

        if scheduler_use:
            scheduler.step(test_loss)

        # save the model
        torch.save(classfier.state_dict(), weight_path)
        print("SVM model saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--real_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--fake_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--train',type=str, default='train', help='train or val')
    parser.add_argument('--weight_path', type=str, default='./pretrained_weights/svm_model.pth', help='path to save weight')
    parser.add_argument('--optimizer', type=str, default="SGD", help='the optimizer')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--nf', type=int, default=1024, help='number of features')

    parser.add_argument('--arch', type=str, default='CLIP:ViT-L/14')
    parser.add_argument('--classifier', type=str, default='SVM')
    parser.add_argument('--load_ckpt', type=bool, default=False)
    parser.add_argument('--ckpt', type=str, default='./pretrained_weights/fc_weights.pth')

    parser.add_argument('--result_folder', type=str, default='result', help='')
    parser.add_argument('--batch_size', type=int, default=64)

    # Augmentation options
    parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
    parser.add_argument('--cropSize', type=int, default=224, help='then crop to this size')
    parser.add_argument('--rz_interp', default=['bilinear'])
    parser.add_argument('--blur_prob', type=float, default=0.5)
    parser.add_argument('--blur_sig', default=[0.0,0.15])
    parser.add_argument('--jpg_prob', type=float, default=0.5)
    parser.add_argument('--jpg_method', default=['cv2','pil'])
    parser.add_argument('--jpg_qual', default=[85,90,95,100])
       


    opt = parser.parse_args()

    lr = opt.lr   
    #scheduler
    scheduler_use = True
    scheduler_patience = 5
    
    if os.path.exists(opt.result_folder):
        shutil.rmtree(opt.result_folder)
    os.makedirs(opt.result_folder)

    model = get_model(opt.arch)

    if opt.load_ckpt:
        print("Loading Checkpoint", opt.ckpt)
        state_dict = torch.load(opt.ckpt, map_location='cpu', weights_only=True)
        model.fc.load_state_dict(state_dict)

    print ("Model loaded..")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.cuda()

    print("Data Mode", opt.data_mode)

    if(opt.classifier == 'SVM'):
        classifier = SVM(opt.nf,2)
    elif(opt.classifier == 'Linear'):
        model.project_feats = True 
        classifier = LinearProbe(opt.nf,num_classes=1)


    classifier.cuda()
    optimizer = optim.Adam(classifier.parameters(),lr=lr)
    if scheduler_use:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='min',factor=0.1,patience=scheduler_patience, eps=1e-20)

    # if logging:
    #     run = wandb.init(
    #         # Set the project where this run will be logged
    #         project="742 Fake AI Project SVM",
    #         # Track hyperparameters and run metadata
    #         config={
    #             "learning_rate": lr,
    #             "batch_size": opt.batch_size,
    #             "optimizer": optimizer,
    #             "scheduler": scheduler,
    #             "loss_func": "cross_entropy"
    #         },
    #     )

    dataset_path = dict(real_path=opt.real_path, fake_path=opt.fake_path, data_mode=opt.data_mode)

    if (opt.real_path == None) or (opt.fake_path == None) or (opt.data_mode == None):
        loading_mode = "multiple"
    else:
        #dataset_path = [ dict(real_path=opt.real_path, fake_path=opt.fake_path, data_mode=opt.data_mode) ]
        loading_mode = "single"
    
    print("Preparing Training data")
    train_dataset = RealFakeDatasetTraining(dataset_path['real_path'], 
                                            dataset_path['fake_path'], 
                                            dataset_path['data_mode'], 
                                            opt.arch,
                                            loading_mode=loading_mode,
                                            models=REDUCED_PATHS,
                                            opt=opt
                                            )
    print("Preparing Validation data")
    val_dataset = RealFakeDatasetTraining(  dataset_path['real_path'], 
                                            dataset_path['fake_path'], 
                                            dataset_path['data_mode'], 
                                            opt.arch,
                                            train=False,
                                            loading_mode=loading_mode,
                                            models=REDUCED_PATHS,
                                            opt=opt
                                            )

    loaders = {}
    loaders["train_loader"] = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
    loaders["val_loader"] = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    train(classifier, model, loaders, optimizer, scheduler, logging=False, weight_path=opt.weight_path)
