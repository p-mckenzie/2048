# control random processes
import torch
torch.manual_seed(1)
import numpy as np
np.random.seed(1)

from torch import nn
from torch import optim
import time
from collections import defaultdict

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
		
	def random_init(self, max_tile=None):
		# generates a random initial layout, possibly given a max tile for the layout to build off of
		## ex: max tile=64 will generate a 4x4 layout where the largest tile is a 64, all other tile values are smaller
		if max_tile:
			max_tile = int(np.log2(max_tile))
			assert max_tile in range(0,10) # maximum tile cannot be larger than 512
		else:
			max_tile = np.random.choice(range(3,10))
		goal_tile = 2**(max_tile+1)

		# insert the max tile into the matrix
		init_layout = np.zeros((4,4), dtype=np.int)
		init_layout[np.random.choice(range(4)), np.random.choice(range(4))] = 2**max_tile

		for move in range(np.random.choice(range(4,12))):
			row_idx = np.random.choice(np.where((init_layout==0).sum(axis=1)>0)[0])
			init_layout[row_idx, np.random.choice(np.where(init_layout[row_idx]==0)[0])] = 2**np.random.choice(range(1,max_tile))

		return init_layout, goal_tile

	def compute_game_penalties(self, data, penalty_type):
		# all weights range [-1,1], where weight<0 indicates a "bad" game and weight>=0 indicates a "good" game
		if penalty_type=='scores':
			# computed by overall game score
			rank_values = data.final_scores
		elif penalty_type=='max':
			# computed by max tile on board at the end of game
			rank_values = data.max_tile
		elif penalty_type=='log2_max':
			# computed by the base 2 log of max tile on board at the end of game        
			rank_values = np.log2(data.max_tile)
		elif penalty_type=='tile_sums':
			# computed by sum of tile values at the end of game
			rank_values = data.tile_sums
		else: # use binary as default (-1:final score was below median, 1: final score was above median)
			penalties = np.ones(data.final_scores.shape)
			penalties[data.final_scores<=np.median(data.final_scores)] = -1
			return np.nan_to_num(penalties)
		
		# runs for all except default (binary)
		# using distance to median by whatever metric was chosen using penalty_type
		if data.wins.max():
			# distribute between [-1,-.5] for "bad" and [.5, 1] for "good" 
			final_penalties = np.ones(data.final_scores.shape)
			for unique_val in np.unique(data.wins):
				sub_rank = rank_values[data.wins==unique_val]
				penalties = np.nan_to_num((sub_rank-sub_rank.min())/(sub_rank.max()-sub_rank.min())/2)+.5
				if not unique_val:
					penalties *= -1
				final_penalties[data.wins==unique_val] = penalties
		else: # no games won
			# normalize between -1 and 1
			maxes = np.repeat(rank_values.max(), rank_values.shape)
			maxes[rank_values<=np.median(rank_values)] = rank_values.min()
			maxes = np.absolute(maxes-np.median(rank_values))
			final_penalties = (rank_values-np.median(rank_values))/maxes
		return np.nan_to_num(final_penalties)

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

	def get_data(self, init_layout=None, goal_tile=2048, batch_size=10, random_frac=None, 
				 random_games=0, randomized_move=True):
		
		# define method for training (possibly including random moves with neural network-selected moves)
		if random_frac is not None:
			method = lambda layout: self.model(torch.from_numpy(transform(layout)).double().reshape(1,-1)).detach().numpy().flatten() if np.random.random()>random_frac else np.repeat(.25, 4)
		else:
			method = lambda layout: self.model(torch.from_numpy(transform(layout)).double().reshape(1,-1)).detach().numpy().flatten()
		
		# initialize games class
		from helper import GameDriver
		data = GameDriver()
		
		# default option is to run entire games
		if init_layout is not None:
			# run neural-network-run games with initialization
			data.run_games(batch_size, method=method, init_layout=init_layout,
						  early_stop=goal_tile, randomized_move=randomized_move)
			if random_games>0: # run same number of completely random games, if applicable
				data.run_games(random_games, init_layout=init_layout,
							  early_stop=goal_tile, randomized_move=randomized_move) 
		else:
			data.run_games(batch_size, method=method, early_stop=goal_tile, 
						   randomized_move=randomized_move) # run neural-network-run games
			if random_games>0:
				data.run_games(random_games, early_stop=goal_tile, 
							   randomized_move=randomized_move) # run some number of completely random games, if applicable

		return data
	
	def compute_y_yhat(self, data, game_penalty_type, move_penalty_type):
		# ------------------ use data to update model weights ------------------------                            
		# find weights for good/bad game performance and weights for move importance
		game_penalties = self.compute_game_penalties(data, game_penalty_type)
		best = np.argmax(game_penalties)
		move_penalties = self.compute_move_penalties(data, move_penalty_type)
		expanded_game_penalties = np.repeat(game_penalties, data.num_moves)

		# set up data to train
		L = torch.from_numpy(transform(np.concatenate(data.layouts))).double()
		M = torch.from_numpy(np.concatenate(data.moves)).double()
		y_hat = self.model(L)

		# make "true" labels - M where the game was "good", (1-M)/3 where the game was "bad"
		y = M.clone()
		mask = expanded_game_penalties<0
		mask = torch.from_numpy(mask).nonzero().flatten()
		y[mask,:] = (1-y[mask,:])/3        

		# align penalties for torch compatibility
		## note absolute value for game penalties, since directionality was taken care of when generating true labels
		expanded_game_penalties = torch.abs(torch.from_numpy(expanded_game_penalties)[:,None].double())
		move_penalties = torch.from_numpy(move_penalties)[:,None].double()
		
		return expanded_game_penalties*move_penalties*y, expanded_game_penalties*move_penalties*y_hat, best
	
	def train(self, lr=0.001, duration=1/600, random_games=False, random_frac=None, batch_size=10,
			 move_penalty_type=None, game_penalty_type=None, game_type='full',
			 test=False, num_games=2000, memory=.9):
		
		# save all parameters passed to train() in the model object for later serialization
		self.model.user_parameters = locals().copy()
		self.model.user_parameters.pop('self')
		
		# initialize optimizer and loss function
		opt = optim.Adam(self.model.parameters(), lr=lr)
		loss = nn.L1Loss()
		
		# initialize variables to hold data during training
		end_time = time.time()+60*60*duration
		self.model.scores = defaultdict(list)
		
		if game_type=='mini_iterative' or game_type=='mini_random':
			max_idx_start=3
		
		if not test:
			# get baseline performance before training at all
			self.baseline(num_games, save=False)
			
		print_count = 0
		
		while time.time()<end_time: # run loop for a certain duration (in hours)
			
			# ------------------ get appropriate game data ------------------------
			if game_type=='full':
				data = self.get_data(batch_size=batch_size, random_frac=random_frac, random_games=random_games)
				
			else:
				if game_type=='mini_random':
					
					# choose a max tile for random initial layout (draw from uniform distribution)
					random_idx_start = np.random.choice(range(3,max_idx_start+1))
					init_layout, goal_tile = self.random_init(max_tile=2**random_idx_start)
					
					# run mini games with random initial layout
					data = self.get_data(init_layout, goal_tile, batch_size, random_frac, random_games)
					
					# if at least 60% of games were successfully played
					## AND we randomly drew the largest possible index, extend range of indexes available
					## (limited to 10, meaning we can never initialize with a tile greater than 1048)
					if data.wins.mean()>.6 and max_idx_start==random_idx_start:
						max_idx_start = min(10, max_idx_start+1)
				
				elif game_type=='mini_iterative':
					
					if max_idx_start==3: # need to start with a random layout (new round)
						init_layout, goal_tile = self.random_init(max_tile=2**max_idx_start)
					
					for attempt in range(10):
						# run mini games with random layout OR layout left over from last pass
						#return init_layout, goal_tile, batch_size, random_frac, random_games
						data = self.get_data(init_layout=init_layout, goal_tile=goal_tile, batch_size=batch_size, 
											 random_frac=random_frac, random_games=random_games)
						
						if data.wins.max(): # if at least one mini game was successful
							# set up initial layout for next pass 
							## (increment maximum tile, choose appropriate layout for initialization)
							max_idx_start = max(3, (max_idx_start+1)%11)
							break
							
						if attempt==9: # couldn't make it to next tile from this layout after 10 attempts
							max_idx_start = 3 # start a new round
							
			
			# log performance of training epoch
			for i in range(3,12):
				self.model.scores[2**i].append(np.count_nonzero(data.max_tile==2**i))            
										
			# call function to compute the inputs to our loss function
			new_weighted_y, new_weighted_yhat, best_game = self.compute_y_yhat(data, game_penalty_type, move_penalty_type)
			
			if game_type=='mini_iterative':
				# set up initial layout for next pass 
				## (choose appropriate layout for initialization)        
				if max_idx_start>3:
					init_layout = data.final_layouts[best_game]
					goal_tile = 2**(max_idx_start+1)
			
			try:
				# sample memory% of existing (training data from previous batches)
				indx = torch.randperm(weighted_y.shape[0])[:int(weighted_y.shape[0]*memory)]
				
				# add full new batch
				weighted_y = torch.cat((weighted_y[indx, :], new_weighted_y), 0)
				weighted_yhat = torch.cat((weighted_yhat[indx, :], new_weighted_yhat), 0)

			except NameError:
				weighted_y = new_weighted_y
				weighted_yhat = new_weighted_yhat
				
			# update model weights
			output = loss(weighted_y,
						 weighted_yhat)
			output.backward(retain_graph=True)
			opt.step()
			opt.zero_grad()
			
			if print_count%100==0:
				print(round(list(self.model.parameters())[0].sum().item(), 4), 
				  round(weighted_yhat.std().item(), 4), 
				  weighted_y.shape[0])
				
				
			print_count+=1
			
		if test:
			return
		
		# run large set of full games to gauge any performance improvement
		self.baseline(num_games)
		
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