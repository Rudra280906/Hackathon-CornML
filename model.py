import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import math

import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import rasterio
import os
import glob
import warnings
import random
import numpy as np
import torchdata.stateful_dataloader
import csv
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

    @property
    def classes(self):
        return self.data.classes

class YieldPerAcre(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.base_model = timm.create_model("efficientnet_b0", pretrained=True)

        enet_out_size = 1280
        self.classifier = nn.Linear(enet_out_size, num_classes)
    
    def forward(self, x):
        output = self.classifier(x)
        return output

class CSVModel(nn.Module):
    def __init__(self, in_features=7, h1=8, h2=9, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

torch.manual_seed(41)
# Model for CSV
model = CSVModel()
data_path = "../2022/2022/DataPublication_final/GroundTruth/HYBRID_HIPS_V3.5_ALLPLOTS.csv"
my_df = pd.read_csv(data_path, skiprows=135, usecols=(3, 5, 7, 8, 9, 10, 11, 17), names=["irrigationProvided", "poundsofNitrogenPerAcre", "plotLength", "block", "row", "range", "plotNumber", "yieldPerAcre"])
my_df = my_df.astype(dtype=np.float32)
X = my_df.drop('yieldPerAcre', axis=1)
Y = my_df['yieldPerAcre']
X = X.values
Y = Y.values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=41)

X1 = preprocessing.LabelEncoder()
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
Y_train = torch.from_numpy(Y_train)
# Y_train = Y_train.type(torch.LongTensor)
Y_test = torch.from_numpy(Y_test)
# Y_test = Y_test.type(torch.LongTensor)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 50
losses = []
for i in range(epochs):
    y_pred = model(X_train)
    Y_train_1 = Y_train.reshape((1725, 1))
    loss = criterion.forward(y_pred, Y_train_1)
    losses.append(loss.detach().numpy())
    if i % 10 == 0:
        print(f'Epoch: {i} and loss: {loss}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# [i*int(math.floor(1725/epochs)):(i+1)*int(math.floor(1725/epochs))]