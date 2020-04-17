import numpy as np
np.random.seed(1)

class MonteCarloGameDriver():    
    def __init__(self):
        self.default_moves = np.array(['w','a','s','d'])
        self.probability_distribution = np.array([.25,.25,.25,.25])

    def run_game(self, simulation_size=20):
        from game import GameLayout
        from copy import deepcopy
        
        game = GameLayout()
        
        while game.active:
            # simulate simulation_size games starting at this point
            game_performance = self.simulate(game, simulation_size)
            
            if len(game_performance)==0:
                game.end_game()
                
                print("After {} simulations, achieved max tile {} and score {}".format(simulation_size, game.final_layout.max(), game.score))
                break

            # return the first move with highest average score
            recommendation = max(game_performance, key=game_performance.get)
    
            game.swipe(recommendation)
                
        # game is over
        self.log_game(game)

            
    def simulate(self, game, simulation_size):
        from collections import defaultdict
        game_performance = defaultdict(list)
        
        from copy import deepcopy

        for i in range(simulation_size):
            # run copy game multiple times, saving final scores and first moves each time
            game_copy = deepcopy(game)
            game_copy.reset()

            while game_copy.active:
                move_order = self.weighted_shuffle(self.default_moves, self.probability_distribution)
                for move in move_order:
                    try:
                        game_copy.swipe(move)
                        break
                    except:
                        # move didn't work, try next move
                        continue
            # log final score and first move
            try:
                game_performance[self.default_moves[(game_copy.moves[0]==1).argmax()]].append(game_copy.score)
            except AttributeError:
                pass
            
        # get average score for each first move
        game_performance = {key: np.mean(val) for key, val in game_performance.items()}
            
        return game_performance

    
    def weighted_shuffle(self, options,weights):
        lst = list(options)
        w = [None]*len(lst) # make a copy
        for i in range(len(lst)):
            win_idx = np.random.choice(range(len(lst)), p=weights)
            w[i] = lst[win_idx]
            del lst[win_idx]
            weights = np.delete(weights, win_idx)
            weights = weights/weights.sum()
        return w


    def log_game(self, game):
        assert not game.active # must be a finished game
        try:
            self.final_scores = np.append(self.final_scores, game.score)
            self.num_moves = np.append(self.num_moves, game.num_moves)
            self.layouts.append(game.layouts)
            self.final_layouts.append(game.final_layout)
            self.moves.append(game.moves)
            self.scores.append(game.scores)
            self.tile_sums = np.append(self.tile_sums, game.final_layout.sum())
            self.max_tile = np.append(self.max_tile, game.final_layout.max())
            self.wins = np.append(self.wins, game.won)

        except AttributeError:
            self.final_scores = np.array(game.score)
            self.num_moves = np.array(game.num_moves)
            self.layouts = [game.layouts]
            self.final_layouts = [game.final_layout]
            self.moves = [game.moves]
            self.scores = [game.scores]
            self.tile_sums = np.array(game.final_layout.sum())
            self.max_tile = np.array(game.final_layout.max())
            self.wins = np.array(game.won)