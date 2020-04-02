# control random processes
import torch
torch.manual_seed(1)
import numpy as np
np.random.seed(1)

from torch import nn
from torch import optim
import time

def transform(layout):
    return np.log2(np.where(layout==0, 1, layout))/11-.25

class NeuralNetwork():
    def __init__(self, inputSize=16, outputSize=4, neuronCountJ=50, neuronCountK=100):
        # initialize network
        self.model = nn.Sequential(nn.Linear(inputSize, neuronCountJ),
                       nn.ReLU(), 
                       nn.Linear(neuronCountJ, neuronCountK),
                       nn.ReLU(),
                       nn.Linear(neuronCountK, outputSize),
                       nn.Softmax(dim=1),
                     )
        self.model.double()

    def get_rank_values(self, data, rank_type):
        # all weights range [-1,1], where weight<0 indicates a "bad" game and weight>=0 indicates a "good" game
        if rank_type=='max':
            # computed by max tile on board at the end of game
            rank_values = data.max_tile
        elif rank_type=='log2_max':
            # computed by the base 2 log of max tile on board at the end of game        
            rank_values = np.log2(data.max_tile)
        elif rank_type=='tile_sums':
            # computed by sum of tile values at the end of game
            rank_values = data.tile_sums
        else: # rank by final scores
            rank_values = data.final_scores
        return np.nan_to_num(rank_values)

    def compute_move_penalties(self, data, penalty_type):
        # all weights range [0,1]
        if penalty_type=='nonzero':
            # weights by fraction of tiles that are nonzero
            weights = np.count_nonzero(np.concatenate(data.layouts), axis=1)/16
        elif penalty_type=='linear_move_num':
            # weights linearly by move number (move #/total # of moves)
            weights = np.concatenate([np.linspace(0,1,num_moves) for num_moves in data.num_moves])
        elif penalty_type=='exponential_move_num':
            # weights exponentially (1-e^(-3x)) by move number where x=(move #/total # of moves)
            weights = np.concatenate([1-np.exp(-3*np.linspace(0, 1, num=num_moves)) for num_moves in data.num_moves])
        else:
            # weight all moves equally
            weights = np.ones(data.num_moves.sum())
        return weights

    def get_data(self, batch_size=10, random_frac=None, 
                 random_games=0, randomized_move=True):

        # define method for training (possibly including random moves with neural network-selected moves)
        if random_frac is not None:
            method = lambda layout: self.model(torch.from_numpy(transform(layout)).double().reshape(1,-1)).detach().numpy().flatten() if np.random.random()>random_frac else np.repeat(.25, 4)
        else:
            method = lambda layout: self.model(torch.from_numpy(transform(layout)).double().reshape(1,-1)).detach().numpy().flatten()

        # initialize games class
        from helper import GameDriver
        data = GameDriver()

        # run entire games
        # run neural-network-run games with initialization
        data.run_games(batch_size, method=method, randomized_move=randomized_move)
        if random_games>0: # run same number of completely random games, if applicable
            data.run_games(random_games) 
        
        return data

    def train(self, lr=0.001, duration=1/600, random_games=False, random_frac=None, batch_size=10,
             move_penalty_type=None, game_penalty_type=None,
             test=False, num_games=2000, forget_after=30):

        # save all parameters passed to train() in the model object for later serialization
        self.model.user_parameters = locals().copy()
        self.model.user_parameters.pop('self')

        # initialize optimizer and loss function
        opt = optim.Adam(self.model.parameters(), lr=lr)
        loss = nn.L1Loss()

        # initialize variables to hold data during training
        end_time = time.time()+60*60*duration
        
        self.model.evaluation = []
        '''
        if not test:
            # get baseline performance before training at all
            self.baseline(num_games, save=False)
        '''
        while time.time()<end_time: # run loop for a certain duration (in hours)

            # ------------------ get appropriate game data ------------------------
            data = self.get_data(batch_size=batch_size, random_frac=random_frac, random_games=random_games)

            # -----------------update dataset with new games-----------------------
            try:
                # store M (move performed), and L (layouts presented before choosing moves)
                L = torch.cat([L, torch.from_numpy(np.concatenate(data.layouts)).double()])
                M = torch.cat([M, torch.from_numpy(np.concatenate(data.moves)).double()])
                
                # weights for penalizing different moves/games
                game_rank_values = np.concatenate([game_rank_values, self.get_rank_values(data, rank_type=None)])
                game_sizes = np.concatenate([game_sizes, data.num_moves])
                full_move_penalties = torch.cat([full_move_penalties, 
                          torch.from_numpy(self.compute_move_penalties(data, penalty_type=move_penalty_type))[:,None].double()])
                
                # forget old batch, once training set is large enough
                if len(game_sizes)>forget_after*batch_size:
                    move_cutoff = game_sizes[:batch_size].sum()
                    
                    L = L[move_cutoff:, :]
                    M = M[move_cutoff:, :]
                    full_move_penalties = full_move_penalties[move_cutoff:]
                    
                    game_rank_values = game_rank_values[batch_size:]
                    game_sizes = game_sizes[batch_size:]
                    
                    self.model.evaluation.append(game_rank_values.mean())
                    
                
            except NameError:
                # first time, must make dataset
                L = torch.from_numpy(np.concatenate(data.layouts)).double()
                M = torch.from_numpy(np.concatenate(data.moves)).double()
                game_rank_values = self.get_rank_values(data, rank_type=game_penalty_type)
                game_sizes = data.num_moves
                full_move_penalties = torch.from_numpy(self.compute_move_penalties(data, penalty_type=move_penalty_type))[:,None].double()
            
            # ---------------compute penalties for all batches of games---------------------
            # using distance to median by whatever metric was chosen using penalty_type
            # normalize between [-1,1]
            maxes = np.repeat(game_rank_values.max(), game_rank_values.shape)
            maxes[game_rank_values<=np.median(game_rank_values)] = game_rank_values.min()
            maxes = np.absolute(maxes-np.median(game_rank_values))
            final_penalties = np.nan_to_num((game_rank_values-np.median(game_rank_values))/maxes)
            expanded_game_penalties = np.repeat(final_penalties, game_sizes)
            del maxes, final_penalties
            
            
            # ---------------------------compute y and y_hat-----------------------------------
            y_hat = self.model(L)

            # make "true" labels - M where the game was "good", (1-M)/3 where the game was "bad"
            y = M.clone()
            mask = expanded_game_penalties<0
            mask = torch.from_numpy(mask).nonzero().flatten()
            y[mask,:] = (1-y[mask,:])/3        

            # align penalties for torch compatibility
            ## note absolute value for game penalties, since directionality was taken care of when generating true labels
            expanded_game_penalties = torch.abs(torch.from_numpy(expanded_game_penalties)[:,None].double())
            
            weighted_y = expanded_game_penalties*full_move_penalties*y
            weighted_yhat = expanded_game_penalties*full_move_penalties*y_hat
        
        
            # ---------------------------actually train!-----------------------------------
            # update model weights
            output = loss(weighted_y,
                         weighted_yhat)
            output.backward(retain_graph=True)
            opt.step()
            opt.zero_grad()

        if test:
            return

        # run large set of full games to gauge any performance improvement
        #self.baseline(num_games)

        # call log_results to save the trained model
        self.log_results()

    def baseline(self, num_games, save=False):
        data = self.get_data(batch_size=num_games, randomized_move=False)
        results = np.asarray(np.unique(data.max_tile, return_counts=True)).T
        print(results)
        if save:
            self.model.baseline = results
        return

    def log_results(self):
        import os
        directory='.\\model_results'

        # generate folder for models, if necessary
        if not os.path.exists(directory):
            os.makedirs(directory)

        # save model
        model_int = 0
        filename = os.path.join(directory, 'model_{}.pickle'.format(model_int))

        while os.path.isfile(filename):
            model_int += 1
            filename = os.path.join(directory, 'model_{}.pickle'.format(model_int))

        import pickle
        pickle.dump(self.model, open(filename, 'wb'))