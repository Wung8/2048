import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import numpy as np
import random

import glob, pickle, time
import _2048 as game
from network import Network,Loader

replay = False

data = []
for file in glob.glob("data/data2048*"):
    print(file)
    with open(file,'rb') as f:
        data += pickle.load(f)

print(f"{len(data)} samples")

if replay:
    for board,usr in data:
        game.displayBoardPlt(board)
        time.sleep(.1)
        usr = game.INPUT[usr]
        board,score,penalty = game.move(board,usr,display=True)

model = Network()

batch_size = 50
test_size = 200

train_data = Loader(data)
train_dataloader = DataLoader(
    train_data,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 0
    )

falloff = .98
lr = .0005
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.MSELoss()

model.to(device)

for epoch in range(5):
    print(f'\nstarting epoch {epoch+1}')

    for i, (boards, vals) in enumerate(train_dataloader):
        #labels = labels.type(torch.LongTensor)
        boards, vals = boards.to(device), vals.to(device)

        # forwardprop
        outputs = model(boards)
        loss = loss_fn(outputs, vals)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % test_size == 0: print(loss.item())
        

torch.save(model.state_dict(), f'model2048.pt')        
print('finished training')    








