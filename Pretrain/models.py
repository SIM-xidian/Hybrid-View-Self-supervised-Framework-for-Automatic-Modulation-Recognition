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
    

