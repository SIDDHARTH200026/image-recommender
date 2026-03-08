import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(model, loader, save_path='features.npy'):
    features, labels = [], []
    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            feats = model(images).cpu().numpy().reshape(images.size(0), -1)
            features.append(feats)
            labels.extend(lbls.numpy())
    features = np.vstack(features)
    np.save(save_path, features)
    with open('labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    return features, np.array(labels)

def train_head(features, labels, epochs=3):
    model = nn.Linear(features.shape[1], 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    features_t = torch.FloatTensor(features).to(device)
    labels_t = torch.LongTensor(labels).to(device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(features_t)
        loss = criterion(outputs, labels_t)
        loss.backward()
        optimizer.step()
        if epoch == epochs-1:
            acc = accuracy_score(labels, outputs.argmax(1).cpu().numpy())
            print(f"  Accuracy: {acc:.3f}")
    
    return model
