from .dino import dino

import torch.nn as nn 

CHANNELS = {
    "vit_b_16" : 768,
}

# python3 validate.py --arch=DINO:vit_b_16 --ckpt=pretrained_weights/fc_weights.pth --result_folder=dino_vitb16
class DINOModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(DINOModel, self).__init__()

        self.model = dino.fetch_pretrained_model(name)

        #self.fc = nn.Linear(CHANNELS[name], num_classes) #manually define a fc layer here
        

    def forward(self, x):
        feature = self.model(x)
        return feature
