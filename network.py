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
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU())
        self.dropout = nn.Dropout(p=0.2)
        
        self.fc1 = nn.Linear(4*4*16+2*2*14*32,200)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(200,4)
        #self.act2 = nn.Sigmoid()
        self.act2 = nn.Softmax(dim=1)

    def forward(self,x):
        x1 = self.layer1(x)
        x1 = torch.flatten(x1,start_dim=1)
        x1 = self.dropout(x1)
        
        x = torch.flatten(x,start_dim=1)
        x = torch.cat((x,x1),dim=1)

        x = self.fc1(x)
        x = self.act1(x)
        
        x = self.fc2(x)
        x = self.act2(x)
        return x

class Loader(torch.utils.data.Dataset):
    
    key = {'w':0,'a':1,'s':2,'d':3}
    key_lst = ['w','a','s','d'] * 2
    def __init__(self,data):
        self.data = []
        for board,move in data:
            for board,move in self.getRotations(board,move):
                board = convBoard(board)
                val = F.one_hot(torch.tensor(move),num_classes=4).type(torch.float)
                self.data.append((board,val))

    def getRotations(self,board,move):
        toreturn = []
        for i in range(4):
            newb = board[:]
            newm = self.key[move]
            for j in range(i):
                newb = self.rotateBoard(newb)
                newm += 1
            toreturn.append((newb,newm%4))
        return toreturn
    
    def rotateBoard(self,board):
        newb = []
        for y in range(4):
            for x in range(4):
                i = 4*x + 3-y
                newb.append(board[i])
        return newb

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        return self.data[index]
        
def convBoard(b):
  board = [b[i*4:i*4+4][:] for i in range(4)]
  board = F.one_hot(torch.tensor(board),num_classes=16)
  board = torch.swapaxes(board,0,2)
  board = torch.swapaxes(board,1,2)
  board = board[None,:,:,:]
  board = board.float()
  return board
