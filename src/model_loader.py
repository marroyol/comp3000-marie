import os

import torch
import torch.nn as nn
import torchvision.models as models

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The Gradio app imports this path and loads the trained checkpoint once at startup
model_path = os.path.join(base_dir, "cat_model_resnet18.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_resnet18_model(num_landmarks=48, pretrained=False):
    """Create a ResNet-18 regressor with one x/y output pair per landmark"""
    output_features = num_landmarks * 2

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, output_features)

    return model

model = get_resnet18_model(num_landmarks=48, pretrained=False)