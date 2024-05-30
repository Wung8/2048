import time

import _2048 as game
from network import Network

import gymnasium as gym
from gymnasium import spaces

rev_key = {0:'w',1:'a',2:'s',3:'d'}

class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode = "human"):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.render_mode = "human"
        self.action_space = spaces.Discrete(4)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=1, shape=(16,4,4))

        self.board = game.beginGame()
        

    def step(self, action):
        
        usr = rev_key[action]
        usr = game.INPUT[usr]
        self.board, reward, penalty = game.move(self.board,usr)
        observation = game.convBoard(self.board).numpy().squeeze()

        done = game.gameOver(self.board)
        
        truncated = False
        info = {}
        
        return observation, reward, done, truncated, info

    def reset(self, **kwargs):
        self.board = game.beginGame()
        observation = game.convBoard(self.board).numpy().squeeze()
        info = {}
        return observation, info

    def render(self, mode='human'):
        game.displayBoardPlt(self.board)
        
    def close(self):
        pass


from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

env = CustomEnv(render_mode="human")
#check_env(env)
env = make_vec_env(lambda: env, n_envs=1)

from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
import torch

class CustomPolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args,**kwargs)

        self.model = Network()
        self.model.load_state_dict(torch.load('model2048.pt',map_location=torch.device('cpu')))

    def forward(self, board, state=None, deterministic=False, **kwargs):
        global inp
        inp = torch.tensor(board.sample(),dtype=torch.float32)[None,None,:,:,:]
        #board = game.convBoard(board)
        return self.model(inp)
    
    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.model(observation)

policy = CustomPolicy('MlpPolicy', env)
agent = PPO(policy=policy, env=env, verbose=1)
agent.learn(500)







            
