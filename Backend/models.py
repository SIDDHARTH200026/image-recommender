import torch.nn as nn
from torchvision.models import resnet50, vgg16, ResNet50_Weights, VGG16_BN_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_feature_extractor(model_name):
    if model_name == 'resnet':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model = nn.Sequential(*list(model.children())[:-1])
    else:  # vgg
        model = vgg16(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        model = nn.Sequential(*list(model.children())[:-1])
    model.to(device)
    model.eval()
    return model
