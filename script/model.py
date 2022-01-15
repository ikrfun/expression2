import torch
import torch.nn as nn
import torch.nn.functional as F





class CNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size = 5, padding = 2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,64,kernel_size = 5,padding = 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(16*16*64,128)
        self.fc2 = nn.Linear(128,4)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x

