import torch.nn.functional as F
from torch.nn import init
from torch import nn
import torch
'''
class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()

        self.fc1 = nn.Linear(4*4*16,500)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(500,4)
        self.act2 = nn.Softmax()

    def forward(self,x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x
'''
class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=36, kernel_size=3),
            nn.ReLU())

        self.fc1 = nn.Linear(4*4*16+2*2*14*36,200)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(200,4)
        self.act2 = nn.Softmax()

        self.fc3 = nn.Linear(200,1)


    def forward(self,x):
        x1 = self.layer1(x)
        x1 = torch.flatten(x1)

        x = torch.cat((torch.flatten(x),x1))

        x = self.fc1(x)
        x = self.act1(x)

        x2 = self.fc3(x)
        
        x = self.fc2(x)
        x = self.act2(x)
        return x,x2

class Loader(torch.utils.data.Dataset):
    def __init__(self,data):
        self.data = []
        for item in data:
            board,val = item[:2]
            val = torch.tensor(val).type(torch.float)
            self.data.append((board,val))
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        return self.data[index]
        
