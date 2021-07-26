import gym
import time
import random
import numpy as np

board_size = 19
go_env = gym.make('gym_go:go-v0', size=board_size, komi=0, reward_method='real')

class Tard_Bot:
    def __init__(self,go_env):
        self.s = np.zeros((board_size,board_size))
        self.go_env = go_env
    def move(self):
        while True:
            x = random.randrange(board_size)
            y = random.randrange(board_size)
            if( self.s[x][y] != 1):
                #print((x,y))
                #print(s[x][y])
                state, reward, done, info = go_env.step((x,y))
                self.s = state[3]
                return self.go_env