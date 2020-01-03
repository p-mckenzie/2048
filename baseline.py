'''
Runs 2000 completely random games and prints the max tile and number of games that achieved that tile.

Expected output:
[[  16    1]
 [  32   71]
 [  64  587]
 [ 128 1056]
 [ 256  284]
 [ 512    1]]
'''

from helper import GameDriver
import numpy as np
np.random.seed(1)

data = GameDriver()
data.run_games(2000)

print(np.asarray(np.unique(data.max_tile, return_counts=True)).T)

