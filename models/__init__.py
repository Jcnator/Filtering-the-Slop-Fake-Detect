from .clip_models import CLIPModel
from .dino_models import DINOModel


VALID_NAMES = [
    'DINO:vit_b_16',
    'CLIP:ViT-L/14', 
]


def get_model(name):
    print("Get Model", name)
    assert name in VALID_NAMES
    if name.startswith("CLIP:"):
        return CLIPModel(name[5:])  
    elif name.startswith("DINO:"):
        return DINOModel(name[5:])  
    else:
        assert False 
