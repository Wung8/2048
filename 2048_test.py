import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import time

import _2048 as game
from network import Network

model = Network()
model.load_state_dict(torch.load('model2048.pt'))

rev_key = {0:'w',1:'a',2:'s',3:'d'}

def play():
    board = game.beginGame()
    total,penalty = 0,0
    while True:
        game.displayBoardPlt(board)
        #time.sleep(.1)
        usr = model(game.convBoard(board)[None,:,:,:])
        actions = torch.sort(usr,descending=True)[1].tolist()[0]
        penalty = 1
        while penalty == 1:
            if len(actions) == 0: return total,board
            usr = rev_key[actions[0]]
            usr = game.INPUT[usr]
            board,score,penalty = game.move(board,usr)
            del actions[0]
        total += score

for i in range(10):
    total,board = play()
    print(f'{total}\t{2**max(board)}')
    
    
    
    
