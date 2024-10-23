import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pk
import numpy as np
from utils import *
from tqdm import tqdm
from signal_data_add import get_signal
from torch.utils.data import  DataLoader
import pywt
import matplotlib.pyplot as plt
from patch_encoder_and_projection_head_and_classifier import *
from pytorch_pretrained_vit import ViT
import os

def get_class(class_name,input_shape,num_classes):
    if class_name == 'CNN':
        return CNN(input_shape,num_classes)
    if class_name == 'MCLDNN':
        return MCLDNN(num_classes)
    if class_name == 'PET':
        return PET(input_shape,num_classes)
    if class_name == 'CLDNN':
        return CLDNN(input_shape,num_classes)

class Model(nn.Module):
    def __init__(self,feature_dim,nh,nl):
        super().__init__()
        self.CWTs_emb = patch_embedding_for_CWTs(feature_dim)
        self.singals_emb = patch_embedding_for_singals(feature_dim)
        self.videos_emb = patch_embedding_for_videos(feature_dim)
        #特征提取器
        self.extractor = ViT(feature_dim,128,num_heads=nh,num_layers=nl)
       
        #layer_norm层（输入）
        self.singals_norm_layer = nn.LayerNorm([2,128])#个体初始化一致性
        self.CWTs_norm_layer = nn.LayerNorm([2,99,128])
        self.video_norm_layer = nn.LayerNorm([128,1,14,14])
        #layer_norm层（特征）
        self.feature_norm_layer = nn.LayerNorm([128,feature_dim]).cuda()#以特征为归一化标准，不能以个体吧？如果以个体，那所有个体都一样了
        
    def forward(self,singals,CWTs,videos):
        #norm
        singals = self.singals_norm_layer(singals)
        CWTs = self.CWTs_norm_layer(CWTs)
        videos = self.video_norm_layer(videos)
 
        singals_embed = self.feature_norm_layer(self.singals_emb(singals))
        CWTs_embed = self.feature_norm_layer(self.CWTs_emb(CWTs))#每一种模态的特征在进入transformer前，都会进行layer norm，以确保特征间的一致性
        videos_embed = self.feature_norm_layer(self.videos_emb(videos))
        #layer_norm以每个模态的一个“个体”为单位（可以更换为以一个特征为单位）
        singals_feature = self.extractor(singals_embed)
        CWTs_feature = self.extractor(CWTs_embed)#这里应该不用加layer norm，因为transformer的最后有norm
        videos_feature = self.extractor(videos_embed)
        return singals_feature,CWTs_feature,videos_feature
class Proj(nn.Module):
    def __init__(self,in_feature_dim,out_feature_dim):
        super().__init__()
        #projection head
        self.singals_proj = prejection_head(in_feature_dim,out_feature_dim)
        self.CWTs_proj = prejection_head(in_feature_dim,out_feature_dim)
        self.videos_proj = prejection_head(in_feature_dim,out_feature_dim)

    def forward(self,singals_feature,CWTs_feature,videos_feature):
        singals_projection = self.singals_proj(singals_feature)
        CWTs_projection = self.CWTs_proj(CWTs_feature)#这里应该不用加layer norm，因为transformer的最后有norm
        videos_projection = self.videos_proj(videos_feature)
        
        return singals_projection,CWTs_projection,videos_projection
    

class CNN(nn.Module):
    def __init__(self,input_shape,num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,5)),nn.BatchNorm2d(32),nn.LeakyReLU(),nn.MaxPool2d(1,2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,3)),nn.BatchNorm2d(32),nn.LeakyReLU(),nn.MaxPool2d(1,2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,3)),nn.BatchNorm2d(32),nn.LeakyReLU(),nn.MaxPool2d(1,2)
        )
        dim1 = (((input_shape[1]-4)//2-1)//2-1)//2*32
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, 100),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(100, num_classes))

    def forward(self, x):
        x = torch.unsqueeze(x,dim=1)
        out = x
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.fc1(out)
        out = self.fc2(out)

        return out
    
class MCLDNN(nn.Module):
    def __init__(self,num_classes):
        super(MCLDNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,50,(2,8)),nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(1, 50, 8), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(1, 50, 8), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(50, 50, (1,8)), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(100, 100, (2,5)), nn.ReLU())

        self.lstm = nn.LSTM(100,128,num_layers=3)

        self.fc1 = nn.Sequential(nn.Linear(128,128),nn.SELU(),nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(128,num_classes),nn.SELU(), nn.Dropout(0.5))

    def forward(self,x):
        input_iq = x.unsqueeze(1)
        input_i = x[:,0,:].unsqueeze(1)
        input_q = x[:,1,:].unsqueeze(1)
        input_iq = self.conv1(input_iq)
        input_iq = F.pad(input_iq, [3,4,0,1], "constant", 0)
        input_i = self.conv2(input_i)
        input_i = F.pad(input_i, [7, 0], "constant", 0)
        input_q = self.conv3(input_q)
        input_q = F.pad(input_q, [7, 0], "constant", 0)
        input_i = input_i.unsqueeze(2)
        input_q = input_q.unsqueeze(2)
        inputicq = torch.cat([input_i, input_q], 2)
        inputicq = self.conv4(inputicq)
        inputicq = F.pad(inputicq, [3, 4, 0, 0], "constant", 0)
        input = torch.cat([input_iq, inputicq], 1)
        input = self.conv5(input)
        input = input.reshape(input.shape)
        input = torch.squeeze(input,dim=2).permute(2, 0, 1)
        input,_ = self.lstm(input)
        input = input[-1, :, :]
        input = self.fc1(input)
        input = self.fc2(input)

        return input
    
class classifier_head_layer_2(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim,in_dim)
        self.fc2 = nn.Linear(in_dim,out_dim)

    def forward(self,x):
        return self.fc2(nn.functional.relu(self.fc1(x)))

class classifier_head_layer_3(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim,in_dim)
        self.fc2 = nn.Linear(in_dim,in_dim)
        self.fc3 = nn.Linear(in_dim,out_dim)


    def forward(self,x):
        return self.fc3(nn.functional.relu(self.fc2(nn.functional.relu(self.fc1(x)))))

class classifier_head_layer_4(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim,in_dim)
        self.fc2 = nn.Linear(in_dim,in_dim)
        self.fc3 = nn.Linear(in_dim,in_dim)
        self.fc4 = nn.Linear(in_dim,out_dim)


    def forward(self,x):
        return self.fc4(nn.functional.relu(self.fc3(nn.functional.relu(self.fc2(nn.functional.relu(self.fc1(x)))))))
    


class PET(nn.Module):
    def __init__(self, input_shape = (2,1024), num_classes=11):
        super(PET, self).__init__()
        # Define layers
        self.input_shape = input_shape
        self.fc1 = nn.Linear(input_shape[0] * input_shape[1], 1)  # Linear layer for the first input
        self.conv1_1 = nn.Conv2d(1, 75, kernel_size=(2, 8), padding='valid')
        self.conv1_2 = nn.Conv2d(75, 25, kernel_size=(1, 5), padding='valid')
        self.gru = nn.GRU(input_size=25, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, input1):
        real = input1[:,0,:]
        image = input1[:,1,:]
        # Flatten and dense layer
        x1 = input1.view(input1.shape[0], -1)  # Flatten to (batch_size, 256)
        x1 = self.fc1(x1)  
        cos_value = torch.cos(x1)  # Shape: (batch_size, 1)
        sin_value = torch.sin(x1)  # Shape: (batch_size, 1)
        sig1 = real * cos_value  + image * sin_value
        sig2 = image * cos_value  + real * sin_value
        sig1 = sig1.unsqueeze(1)
        sig2 = sig2.unsqueeze(1)
        signal = torch.cat([sig1,sig2],dim = 1)
        signal = signal.unsqueeze(1)
        x3 = F.relu(self.conv1_1(signal))
        x3 = F.relu(self.conv1_2(x3))

        # Temporal feature extraction
        x4 = x3.view(x3.size(0), self.input_shape[1]-11, 25)  # Reshape for GRU
        x4, _ = self.gru(x4)

        # Final classification
        x = self.fc(x4[:, -1, :])  # Use the last time step
        return x
    

class CLDNN(nn.Module):
    def __init__(self, input_shape, num_classes, dropout_rate=0.4):
        super(CLDNN, self).__init__()
        self.input_shape = input_shape
        self.conv1d = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=8)  # Conv1D layer
        self.pool = nn.MaxPool1d(kernel_size=2)  # Max pooling layer
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)  # First LSTM layer
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout layer
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)  # Second LSTM layer
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout layer
        self.flatten = nn.Flatten()  # Flatten layer
        dim1 = (self.input_shape[1]-8)*32
        self.fc = nn.Linear( dim1, num_classes)  # Fully connected layer

    def forward(self, x):
        x = self.conv1d(x)  # Shape: (batch_size, 64, 121)
        x = self.pool(x)    # Shape: (batch_size, 64, 60)
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, 60, 64) for LSTM
        x, _ = self.lstm1(x)  # LSTM layer
        x = self.dropout1(x)  # Dropout
        x, _ = self.lstm2(x)  # Second LSTM layer
        x = self.dropout2(x)  # Dropout
        x = self.flatten(x)  # Flatten
        
        x = self.fc(x)  # Output layer
        return x