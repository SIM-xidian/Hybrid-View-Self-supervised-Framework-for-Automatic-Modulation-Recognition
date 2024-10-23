import torch
import torch.nn as nn

class patch_embedding_for_CWTs(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.conv = nn.Conv2d(2,dim,kernel_size = (99,3),padding = (0,1))

    def forward(self,x):
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2) 
        return x

class patch_embedding_for_singals(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.conv = nn.Conv2d(1,dim,kernel_size = (2,3),padding = (0,1))

    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2) 
        return x

class patch_embedding_for_videos(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.conv = nn.Conv3d(1,dim,kernel_size = (3,14,14),padding = (1,0,0))

    def forward(self,x):
        x = x.permute(0,2,1,3,4)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2) 
        return x
    
class patch_embedding_for_images(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.conv = nn.Conv2d(1,dim,kernel_size = (14,14),padding = 0)

    def forward(self,x):
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2).repeat(1, 128, 1) 
        return x

class classifier_head(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim,in_dim)
        self.fc2 = nn.Linear(in_dim,out_dim)

    def forward(self,x):
        return self.fc2(nn.functional.relu(self.fc1(x)))


class prejection_head(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim,in_dim)
        self.fc2 = nn.Linear(in_dim,out_dim)

    def forward(self,x):
        return self.fc2(nn.functional.relu(self.fc1(x)))