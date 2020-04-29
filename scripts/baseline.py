'''
Runs 2000 completely random games and prints the average game score, plus the max tile and number of games that achieved that tile.

Expected output:
1079.66
[[ 16   6]
 [ 32 150]
 [ 64 727]
 [128 982]
 [256 134]
 [512   1]]
'''

from helper import GameDriver
import numpy as np
np.random.seed(1)

data = GameDriver()
data.run_games(2000)
print(round(data.final_scores.mean(), 2))
print(np.asarray(np.unique(data.max_tile, return_counts=True)).T)