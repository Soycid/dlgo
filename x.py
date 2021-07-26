import gym
import time
import random
import numpy as np
from bots import Tard_Bot
board_size = 19
go_env = gym.make('gym_go:go-v0', size=board_size, komi=0, reward_method='real')
done = False
handicap_order = [(3,15),(15,3),(15,15),(3,3),(9,15),(9,3),(3,9),(15,9),(9,9)]
def generate_board(handicap):
    if handicap != None:
        for i in range(handicap):
            state, reward, done, info = go_env.step(handicap_order[i])
            go_env.step(None)
        
generate_board(None)
#game loop
t = Tard_Bot(go_env)
while not done:
    go_env = t.move()
    go_env.render(mode='human')
    time.sleep(.5)
print(done)
go_env.render(mode='terminal')