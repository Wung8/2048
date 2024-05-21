import torch.nn.functional as F

import sys; args = sys.argv[1:]
import math, time, random
import numpy as np
import torch

def setGlobals():
  global H,W,DIRS,DIRLOOKUP,START_IDXS,INPUT

  H,W = 4,4
  DIRS = [1,-1,4,-4]
  
  DIRLOOKUP = [{*DIRS[:]} for i in range(16)]
  for i in [0,1,2,3]: DIRLOOKUP[i].remove(-4)
  for i in [0,4,8,12]: DIRLOOKUP[i].remove(-1)
  for i in [12,13,14,15]: DIRLOOKUP[i].remove(4)
  for i in [3,7,11,15]: DIRLOOKUP[i].remove(1)

  START_IDXS = {-1:[0,4,8,12],
                1:[3,7,11,15],
                -4:[0,1,2,3],
                4:[12,13,14,15]}

  INPUT = {'w':-4,
           'a':-1,
           's':4,
           'd':1}

def convBoard(b):
  board = [b[i*4:i*4+4][:] for i in range(4)]
  board = F.one_hot(torch.tensor(board),num_classes=16)
  board = torch.swapaxes(board,0,2)
  board = torch.swapaxes(board,1,2)
  board = board[None,:,:,:]
  board = board.float()
  return board

def rand(idxs):
  return random.choice(idxs)

def rval():
  return random.choice([1,1,1,1,1,1,1,1,1,2])

def getEmpty(board):
  return [i for i in range(16) if board[i]==0]

def beginGame():
  board = [0 for i in range(16)]
  board[rand(getEmpty(board))] = rval()
  board[rand(getEmpty(board))] = rval()
  return board

def move(b,d):
  board = b[:]
  score,penalty = 0,0
  combined = []
  idxs = START_IDXS[d]
  for j in range(4):
    for i in idxs:
      p = 0
      i += -d*j
      if board[i] == 0: continue
      while (d in DIRLOOKUP[i]) and (board[i+d]==0):
        p = board[i]
        board[i],board[i+d] = 0,board[i]
        i += d
      if (d in DIRLOOKUP[i]) and (board[i+d]==board[i]) and (i+d not in combined):
        board[i],board[i+d] = 0,board[i]+1
        score += 2**board[i+d]
        combined.append(i+d)
      else:
        penalty += p
        

  if board == b: return (board,-1,1)
  
  board[rand(getEmpty(board))] = rval()
  return (board,score,penalty)

def gameOver(board):
  nbrs = getNeighbors(board)
  if sum([i[1] for i in nbrs]) == -4: return True
  return False

def getNeighbors(board):
  tr = []
  for d in DIRS:
    b = move(board,d)
    if b != 1: tr.append(b)
  return tr

def displayBoard(board):
  #print()
  for row in range(H):
    print(' '.join((str(s)+' ')[:2] if s!=0 else '. ' for s in board[row*W:row*W+W]))
  print(board,max(board))
  print()

setGlobals()

if __name__ == "__main__":
  board = beginGame()
  total,penalty = 0,0
  while getNeighbors(board):
    print(f'         {total} {penalty}')
    displayBoard(board)
    usr = INPUT[input()]
    board,score,penalty = move(board,usr)
    total += score

