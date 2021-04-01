#!/usr/bin/env python
# coding: utf-8

# In[460]:


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import MNIST
import torch.optim as optim
import matplotlib.pyplot as plt
from torchsummary import summary
from PIL import Image


# In[461]:


class TorchCNN(nn.Module):
    def __init__(self):
        """
        Initialize a 3-layer CNN.

        Args:
            in_channels: number of features of the input image
            hidden_channels: list of two numbers which are number of hidden features
            out_features: number of features in output layer
        """
        super(TorchCNN, self).__init__()

        self.layers = []

        # input layer
        self.convi = nn.Conv2d(1, 96, kernel_size=7, stride=2, padding = 1)
        self.relui = nn.ReLU()

        self.layers = [self.convi, self.relui]
        # hidden layer 1
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding = 1)
        self.norm1 = nn.LocalResponseNorm(96)
        self.conv1 = nn.Conv2d(96, 256, kernel_size=5, stride=2, padding = 1)
        self.relu1 = nn.ReLU()

        self.layers = self.layers + [self.pool1, self.norm1, self.conv1, self.relu1]

        # hidden layer 2
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm2 = nn.LocalResponseNorm(256)
        self.conv2 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding = 1)
        self.relu2 = nn.ReLU()

        self.layers = self.layers + [self.pool2, self.norm2, self.conv2, self.relu2]

        # hidden layer 3
        self.conv3 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding = 1)
        self.relu3 = nn.ReLU()

        self.layers = self.layers + [self.conv3, self.relu3]

        # hidden layer 4
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding = 1)
        self.relu4 = nn.ReLU()

        self.layers = self.layers + [self.conv4, self.relu4]

        # hidden layer 5
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layers = self.layers + [self.pool5]

        # hidden layer 6
        self.flat1 = nn.Flatten()
        self.lino = nn.Linear(6*6*256, 10)
        self.softo = nn.Softmax(dim=1)

        self.layers = self.layers + [self.flat1, self.lino, self.softo]

#         # hidden layer 7
#         self.lin7 = nn.Linear(4096, 4096)

#         self.layers = self.layers + [self.lin7]
        
#         # output layer
#         self.lino = nn.Linear(4096, 10)
#         self.softo = nn.Softmax(dim=1)
        
#         self.layers = self.layers + [self.lino, self.softo]

    def forward(self, x):
        i=0
        for layer in self.layers:
            x = layer(x)
#             print(type(layer))
#             print(x.size())
#             if type(layer)==torch.nn.modules.conv.Conv2d:
#                 i+=1
#                 print(f'layer {i}, {type(layer)}')
        return x


# In[462]:


model = TorchCNN()
model.layers


# In[463]:


mnist_train = torch.load('data/MNIST/processed/training.pt')
mnist_test = torch.load('data/MNIST/processed/test.pt')


# In[464]:


transform = transforms.Compose([transforms.Resize((224,224))])
train = []
index = 0
for i in range(len(mnist_train[0])//100):
    train.append(transform(mnist_train[0][i*100:i*100+100]))
    
train = torch.stack(train).view(60000,224,224)

train_data = []
for data, label in zip(train, mnist_train[1]):
    train_data.append([data[None,:,:],label])
    
test = []
for i in range(len(mnist_test[0])//100):
    test.append(transform(mnist_test[0][i*100:i*100+100]))
    
test = torch.stack(test).view(10000,224,224)

test_data = []
for data, label in zip(test, mnist_test[1]):
    test_data.append([data[None,:,:],label])

dummy = train_data[0][0][None,:,:,:].clone()
k = dummy.float().clone()

features = []
for i in model.layers:
    k = i(k)
    if k.size()[2] < 224//16:
        break
    features.append(i)
    out_channels = k.size()[1]
print(f'number of layers = {len(features)}')
print(f'number of out channels = {out_channels}')
print()
print(*features, sep = "\n")

faster_rcnn_fe_extractor = nn.Sequential(*features)
faster_rcnn_fe_extractor

training_data = torch.Tensor
for i in range(len(train_data[0])):
    training_data = train

out_map = torch.Tensor(len(train_data),96,1,7,7)
for i in range(len(train_data)):
    out_map[i,:,:,:,:] = faster_rcnn_fe_extractor(train_data)



def generate_anchor_at_point(out_map):
    
    ratios = [0.5, 1, 2]
    anchor_scales = [4, 8, 16, 20]
    anchor_num = len(ratios)*len(anchor_scales)
    
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)

    

    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = subsample * anchor_scales[j] * np.sqrt(ratios[i])
            w = subsample * anchor_scales[j] * np.sqrt(1./ ratios[i])

            index = i * len(anchor_scales) + j

            anchor_base[index, 0] =  - h / 2.
            anchor_base[index, 1] =  - w / 2.
            anchor_base[index, 2] =    h / 2.
            anchor_base[index, 3] =    w / 2.
    
    
    featuremapsize = out_map.size()[-1]
    subsample = 224/featuremapsize
    ctr_x = np.arange(subsample/2, (featuremapsize+.5) * subsample, subsample)
    ctr_y = np.arange(subsample/2, (featuremapsize+.5) * subsample, subsample)
    anchors = np.zeros((len(ctr_x)*len(ctr_y)*anchor_num,4))
    index = 0
    for x in ctr_x:
        for y in ctr_y:
            xy = np.array([y,x,y,x])
            anchors[index:index + anchor_num,:] = anchor_base + xy
            index += anchor_num

    valid_anchors_poses = np.where(((anchors.T[0]>=0) & (anchors.T[1]>=0) &
         (anchors.T[2]<=224) & (anchors.T[3]<=224)),1,0)

    valid_anchors = anchors[valid_anchors_poses>0,:].copy()
    
    return valid_anchors

def boxgen(d):
    while len(d.size())>2:
        d = d.squeeze()
    
    for index, y in enumerate(d):
        if y.max()>0:
            y1 = index; break

    for index, y in enumerate(reversed(d)):
        if y.max()>0:
            y2 = 224-index; break

    for index, y in enumerate(d.T):
        if y.max()>0:
            x1 = index; break

    for index, y in enumerate(reversed(d.T)):
        if y.max()>0:
            x2 = 224-index; break

    anchor = np.array([y1,x1,y2,x2])
    return anchor

training_anchors = torch.Tensor(len(training_data),4)
training_labels = torch.Tensor(len(training_data))

for i in range(len(training_data)):
    training_anchors[i,:] = torch.FloatTensor([boxgen(train_data[i][0])])
    training_labels[i] = torch.LongTensor([train_data[i][1]])
