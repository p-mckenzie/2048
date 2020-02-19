import numpy as np

class GameLayout:
    def __init__(self, early_stop=2048):
        # initialize empty layout and zero points
        self.layout = np.zeros((4,4), dtype=np.int)
        self.score = 0
        self.won = False
        self.early_stop = early_stop

        # each game starts with 2 full tiles
        self.add_random()
        self.add_random()

        # used for tracking when game ends
        self.failed_moves = set()
        self.active = True

        # needed for correct data logging
        self.num_moves = 0

    def add_random(self):
        # randomly choose any empty tile and fills it with a 2 or 4 tile (with 90%, 10% probabilities, respectively)
        layout = self.layout.flatten()
        options = (layout==0)
        if sum(options)!=0:
            layout[np.random.choice(range(16), p=(np.repeat(1,(16))*options)/sum(options))] = np.random.choice([2,4],
                                                                                                              p=[.9, .1])
        self.layout = layout.reshape((4,4))

    def condense(self, line):
        # combines neighboring tiles of the same type, returns new line and score generated
        score_increase = 0
        for ind in range(3):
            if line[ind+1]==line[ind]:
                line[ind] = 2*line[ind]
                score_increase += line[ind]
                line[ind+1:] = np.concatenate([line[ind+2:,], np.zeros(1).astype(int)])
        return line, score_increase

    def swipe(self, choice):
        scores = np.zeros(4, dtype=np.int)
        new_layout = np.zeros((4,4), dtype=np.int)

        if choice=='w':
            score, new_layout = self.swipe_up(scores, new_layout)
        elif choice=='s':
            score, new_layout = self.swipe_down(scores, new_layout)
        elif choice=='d':
            score, new_layout = self.swipe_right(scores, new_layout)
        elif choice=='a':
            score, new_layout = self.swipe_left(scores, new_layout)
        else:
            raise Exception("Invalid input to swipe.")

        if (new_layout!=self.layout).sum()>0: # some tiles were moved so the move is valid

            self.failed_moves = set() # reset failed move counter

            # update the game's score
            self.score += scores.sum()
            
            if self.layout.max()>=self.early_stop:
                self.end_game()
                self.won = True
                return
            else:
                self.log_data(choice) # log data w/ old layout
                self.layout = new_layout # update to new layout

                # include the random next tile
                self.add_random()
                
            self.num_moves += 1
            
            
        else:
            self.failed_moves.add(choice)
            if len(self.failed_moves)==4: # all moves have been tried and game should end
                self.end_game()
            
            # exception means move didn't change the layout, and another input is required
            raise Exception('Not a valid move.') 
    
        assert self.active
    
    def end_game(self):
        self.active = False
        self.final_layout = self.layout
        
    def swipe_up(self, scores, new_layout):
        for i in range(4):
            new_layout[i], scores[i] = self.condense(np.concatenate((self.layout[self.layout[:,i]>0,i], self.layout[self.layout[:,i]==0,i])))
        new_layout = new_layout.T
        return scores.sum(), new_layout

    def swipe_down(self, scores, new_layout):
        for i in range(4):
            new_layout[i], scores[i] = self.condense(np.concatenate((self.layout[self.layout[:,i]==0,i], self.layout[self.layout[:,i]>0,i]))[::-1])
        new_layout = new_layout.T[::-1]
        return scores.sum(), new_layout

    def swipe_right(self, scores, new_layout):
        for i in range(4):
            new_layout[i], scores[i] = self.condense(np.concatenate((self.layout[i, self.layout[i,:]==0], self.layout[i, self.layout[i,:]>0]))[::-1])
        new_layout = new_layout[:,::-1]
        return scores.sum(), new_layout

    def swipe_left(self, scores, new_layout):
        for i in range(4):
            new_layout[i], scores[i] = self.condense(np.concatenate((self.layout[i, self.layout[i,:]>0], self.layout[i, self.layout[i,:]==0])))
        return scores.sum(), new_layout

    def log_data(self, move):
        formatted_move = np.zeros(4)
        formatted_move[['w','a','s','d'].index(move)] = 1

        try:
            self.layouts = np.concatenate((self.layouts, self.layout.reshape(1,-1)))
            self.moves = np.concatenate((self.moves, formatted_move.reshape(1,-1)))
            self.scores = np.append(self.scores, self.score)
        except AttributeError:
            self.layouts = self.layout.reshape(1,-1)
            self.moves = formatted_move.reshape(1,-1)
            self.scores = np.array(self.score)