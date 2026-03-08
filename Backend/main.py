import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from models import get_feature_extractor
from train import extract_features, train_head
from recommend import recommend_similar
from visualize import visualize_recommendations
import matplotlib.pyplot as plt

print("🚀 Starting Image Recommendation Engine...")

# Data setup
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
loader = DataLoader(dataset, 64, shuffle=False)

# Extract features with ResNet & VGG
print("📥 Extracting ResNet features...")
resnet_model = get_feature_extractor('resnet')
resnet_features, labels = extract_features(resnet_model, loader)

print("📥 Extracting VGG features...")
vgg_model = get_feature_extractor('vgg')
vgg_features, _ = extract_features(vgg_model, loader)

# Train classifier heads
print("🎯 Training ResNet head...")
resnet_head = train_head(resnet_features, labels)
print("🎯 Training VGG head...")
vgg_head = train_head(vgg_features, labels)

# Demo recommendation
print("🔍 Generating recommendations...")
top_idx, scores = recommend_similar(resnet_features, 0)
visualize_recommendations(dataset, 0, top_idx)

print("✅ COMPLETE! Check 'results.png'")
print("🌐 Web demo: streamlit run app.py")
