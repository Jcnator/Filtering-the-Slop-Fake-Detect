from .clip import clip 
from PIL import Image
import torch.nn as nn
import numpy as np
import joblib


CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768
}

class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1, project_feats=False):
        super(CLIPModel, self).__init__()
        print("Starting CLIP Model")
        print(" name:", name)
        print(" num classes", num_classes)

        self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class
        #named_layers = dict(self.model.named_modules())
        #print("Model Layers")
        #print(self.model)
        #self.fc = nn.Linear( CHANNELS[name], num_classes )
        self.project_feats = project_feats

        # # Load SVM
        # self.svm_model = joblib.load("./pretrained_weights/svm_model.pkl")
        # self.scaler = joblib.load("./pretrained_weights/scaler.pkl")
 

    def forward(self, x, return_feature=True):
        # print("x", x.shape)
        # print("Return feature", return_feature)
        features = self.model.encode_image(x)
        projected_feats = features["after_projection"]
        unprojected_feats = features["before_projection"]
        feats = features["before ln post"]  
        # print("projected feates", projected_feats.shape)
        # print("unprojected feates", unprojected_feats.shape)
        # return feats, unprojected_feats, projected_feats
        if self.project_feats:
            return projected_feats
        else:
            return unprojected_feats
    
        if return_feature:
            return feats, unprojected_feats, self.svm_predict(unprojected_feats)
        return self.fc(features)
    
    def svm_predict(self, features):
        feats = features.cpu().detach().numpy()
        feats = self.scaler.transform(feats)
        prediction = self.svm_model.predict(feats)
        return prediction


