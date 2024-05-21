import torch.nn.functional as F
from torch.nn import init
from torch import nn
import torch
import torch.optim as optim

import numpy as np
import random

import game
from network import Network,Loader

from joblib import Parallel, delayed
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt


moves = [1,-1,4,-4]
def run(evals=False):
  print('.',end='',flush=True)
  board = game.beginGame()
  total = 0
  steps = 0

  vals = []
  with torch.no_grad():
    while not game.gameOver(board):
      if evals:
        print(f'         {total}')
        game.displayBoard(board)
      output = model(game.convBoard(board))
      actions,q = output[0].tolist(),output[1].item()
      #actions = model(game.convBoard(board))[0].tolist()
      if evals: print(actions)
      for i,nbr in enumerate(game.getNeighbors(board)):
        if nbr[1] == -1: actions[i] = 0
      i = random.choices([0,1,2,3],actions)[0]
      if evals: i = actions.index(max(actions))
      newboard,r,p = game.move(board,moves[i])
      if r == -1: print('.',end='',flush=True)
      vals.append([board,actions,i,r,p,q])
      
      board = newboard
      total += max(r,0)
      steps += 1

    if evals:
      print(f'         {total}')
      game.displayBoard(board)

  max_b = max(board)

  #print(f'         {total}')
  #game.displayBoard(board)

  next_q,cum_r = 0,0
  for val in vals[::-1]:
    board,actions,i,r,p,q = val
    #cum_r = r + falloff*cum_r
    cum_r = r + next_q
    
    optimizer.zero_grad()
    output,v = model(game.convBoard(board))

    grad = [0.,0.,0.,0.]
    grad[i] = -cum_r/actions[i] + [0.,0.,0.,30][i]
    grad = torch.tensor(grad,requires_grad=True)

    output.backward(grad,retain_graph=True)
    
    loss = loss_fn(v,torch.tensor([cum_r]).float())
    loss.backward()
    optimizer.step()

    #cum_r = r
    next_q = q


  return (total,max_b,steps)


model = Network()

falloff = .9
lr = .0005
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
loss_fn = torch.nn.HuberLoss()

#model.load_state_dict(torch.load('2048-model_20.pt',map_location=torch.device('cpu')))

epochs = 1000
episodes = 50
scores = []
for j in range(0,epochs):
  print(f'epoch {j}|',end=' ')
  #run(evals=True)

  newvals = zip(*Parallel(n_jobs=-2)(delayed(run)() for ep in range(episodes)))
  newvals = [sum(x)/episodes for x in newvals]
  scores.append(newvals[0])
  print(newvals)
  
  #totals = [0,0,0]
  
  #for i in range(iters):
  #  score,max_b,steps = run()
  #  #if i%10==0: print(score,max_b,steps)
  #  totals = [totals[0]+score,totals[1]+max_b,totals[2]+steps]
  #scores.append(totals[0]/iters)
  #print(' '.join([str(i/iters)[:6] for i in totals]))
  #torch.save(model.state_dict(), f'2048-model_{j}.pt')

x = np.array([*range(epochs)])
y = np.array(scores)

a,b = np.polyfit(x,y,1)
print(a)
plt.scatter(x,y)
plt.plot(x,a*x+b)
plt.show()

