#ASSUMPTION: 19x19 GAMES ONLY

from sgfmill import sgf
import gym
import time
import random
import numpy as np
import pandas as pd

X = []
y = []


with open("./test_games/2019-04-01-14.sgf", "rb") as f:
    game = sgf.Sgf_game.from_bytes(f.read())
winner = game.get_winner()
board_size = game.get_size()
root_node = game.get_root()
b_player = root_node.get("PB")
w_player = root_node.get("PW")
state = []
print("Board size:",board_size)
print("Handicap:",game.get_handicap())
go_env = gym.make('gym_go:go-v0', size=board_size, komi=0, reward_method='real')
done = False
handicap_order = [(3,15),(15,3),(15,15),(3,3),(9,15),(9,3),(3,9),(15,9),(9,9)]
last_state = np.zeros((6,19,19))

def pair_to_grid(pair):
    x = np.zeros((19,19))
    x[pair[0]][pair[1]] = 1
    return x

def generate_board(handicap):
    if handicap != None:
        for i in range(handicap):
            last_state, reward, done, info = go_env.step(handicap_order[i])
            go_env.step(None)
generate_board(None)

for node in game.get_main_sequence():
    if node.get_move() != (None,None):
        move = node.get_move()[1]
        state, reward, done, info = go_env.step(move)
        X.append(last_state)
        y.append(pair_to_grid(move))
        last_state = state

