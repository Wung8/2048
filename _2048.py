import sys; args = sys.argv[1:]
import math, time, random
from random import choice as rand
import numpy as np
import cv2
import pickle
import keyboard
from functools import partial
import torch.nn.functional as F
import torch


def setGlobals():
  global H,W,DIRS,DIRLOOKUP,START_IDXS,INPUT,REV_INPUT

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

  INPUT = {'w':-4,'a':-1,'s':4,'d':1,'up arrow':-4,'left arrow':-1,'down arrow':4,'right arrow':1}
  REV_INPUT = {-4:'w',-1:'a',4:'s',1:'d'}


def convBoard(b):
  board = [b[i*4:i*4+4][:] for i in range(4)]
  board = F.one_hot(torch.tensor(board),num_classes=16)
  board = torch.swapaxes(board,0,2)
  board = torch.swapaxes(board,1,2)
  board = board[None,:,:,:]
  board = board.float()
  return board

def rval():
  return random.choice([1,1,1,1,1,1,1,1,1,2])

def getEmpty(board):
  return [i for i in range(16) if board[i]==0]

def beginGame():
  board = [0 for i in range(16)]
  board[rand(getEmpty(board))] = rval()
  board[rand(getEmpty(board))] = rval()
  return board

def move(b,d,display=False):
  board = b[:]
  score,penalty = 0,0
  combined = []
  idxs = START_IDXS[d]
  move_by = [0 for i in range(16)]
  for j in range(4):
    for i in idxs:
      #p = 0
      c = 0
      i += -d*j
      p = i
      if board[i] == 0: continue
      while (d in DIRLOOKUP[i]) and (board[i+d]==0):
        board[i],board[i+d] = 0,board[i]
        i += d
        c += 1
      if (d in DIRLOOKUP[i]) and (board[i+d]==board[i]) and (i+d not in combined):
        board[i],board[i+d] = 0,board[i]+1
        score += 2**board[i+d]
        combined.append(i+d)
        c += 1
      move_by[p] = c

  if board == b: return (board,0,1)

  frames = 4
  if display:
      #print(move_by)
      for i in range(frames):
          displayBoardPlt(b,move_by,d,t=i/frames)
  
  board[rand(getEmpty(board))] = rval()
  return (board,score,0)

def gameOver(board):
  nbrs = getNeighbors(board)
  if sum([i[1] for i in nbrs]) == -4: return True
  return False

def getNeighbors(board):
  tr = []
  for d in DIRS:
    b = move(board,d)
    if b[1] != -1: tr.append(b)
  return tr

def displayBoard(board):
  for row in range(H):
    print(' '.join((str(s)+' ')[:2] if s!=0 else '. ' for s in board[row*W:row*W+W]))
  print(board,max(board))
  print()

def displayBoardPlt(board, move_by=[0 for i in range(16)], d=0, t=0):
    
    scale = 150
    
    colors = [[100,100,100],    # -
            [238, 228, 218],    # 2
            [237, 224, 200],    # 4
            [242, 177, 121],    # 8
            [245, 149, 99],    # 16
            [246, 124, 95],    # 32
            [246, 94, 59],    # 64
            [237, 207, 114],    # 128
            [237, 204, 97],    # 256
            [237, 200, 80],    # 512
            [237, 197, 63],    # 1024
            [237, 194, 46],    # 2048
            [17, 27, 37]]    # 4096
    
    font_colors = [(50,50,50),(240,240,240)]
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = 2.5
    thickness = 7

    def drawTile(img,v,x,y):
        m = move_by[x+4*y]
        if abs(d) == 1: x += t*m*[-1,1][d>0]
        else: y += t*m*[-1,1][d>0]
        
        start = (int(scale*(x+.05)),int(scale*(y+.05)))
        end = (int(scale*(x+.95)),int(scale*(y+.95)))
        cv2.rectangle(img, start, end, colors[min(v,12)], -1)

        val = 2**v
        org = (int(scale*(x+.5)-scale*len(str(val))/6), int(scale*(y+.5)+scale/6))
        color = font_colors[val >= 8]
        image = cv2.putText(img, str(val), org, font,  
                           fontScale, color, thickness, cv2.LINE_AA)

    #img = [[],[],[],[]]
    #for x in range(4):
    #    for y in range(4):
    #        val = board[x+4*y]
    #        img[y].append(colors[val])
    #img = np.array(img,dtype=np.uint8)
    
    img = np.array([[colors[0]]],dtype=np.uint8)
    img = img.repeat(scale*4,axis=0).repeat(scale*4,axis=1)

    for x in range(4):
        for y in range(4):
            val = board[x+4*y]
            if val == 0: continue
            drawTile(img,val,x,y)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


    #for x in range(4):
    #    for y in range(4):
    #        val = 2**board[x+4*y]
    #        if val == 1: continue
    #        org = (int(scale*(x+.5)-scale*len(str(val))/6), int(scale*(y+.5)+scale/6))
    #        color = font_colors[val >= 8]
    #        image = cv2.putText(img, str(val), org, font,  
    #                           fontScale, color, thickness, cv2.LINE_AA)

    
    cv2.imshow("img", img)
    cv2.waitKey(5)
    
            

setGlobals()

to_move = 0
def keyboardInput(k):
    global to_move
    to_move = INPUT[k]

def save():
    global hist,total
    if total < 15000: return
    try:
        with open('data2048', 'rb') as f:
            data = pickle.load(f)
    except:
        data = []
        
    with open('data2048', 'wb') as f:
        pickle.dump(data + hist, f)
    hist = []
    

if __name__ == "__main__":

    for k in ["down arrow","up arrow","left arrow","right arrow",'w','a','s','d']: keyboard.add_hotkey(k, partial(keyboardInput, k))
    keyboard.add_hotkey('space',save)
      
    while True:
        print('Starting Game')
        hist = []
        board = beginGame()
        total,penalty = 0,0
        displayBoardPlt(board)
        while getNeighbors(board):
            time.sleep(.1)
            usr = to_move
            if usr == 0: continue
            print(f'{total} {penalty}')
            usr = to_move
            to_move = 0
            b = board[:]
            board,score,penalty = move(board,usr,display=True)
            total += score
            if penalty == 0:
                hist.append((b,REV_INPUT[usr]))
            displayBoardPlt(board)

        if total > 15000:
            try:
                with open('data2048', 'rb') as f:
                    data = pickle.load(f) 
            except:
                data = []
                
            with open('data2048', 'wb') as f:
                pickle.dump(data + hist, f)








