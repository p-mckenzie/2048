import numpy as np

def print_pretty(layout):
    from math import log2
    def parse_color(cell_value):
        if cell_value>0:
            colors = ["#EEE4DA", "#ECE0CA", "#F2B179", "#F59565", "#F57C5F", "#F65D3B", "#EDCE71", "#EDCC63", "#EDC850", "#EDC53F", "#EEC22E", "#3E3933"]
            return colors[int(log2(cell_value))-1]
        else:
            return "#CDC0B4"

    def cell_block(cell_value):
        return '''
        <td style=font-size:{}px;background-color:#BBADA0;text-align:center;width:100%>
            <div style=color:#776E65;background-color:{};width:60px;height:60px;text-align:center><strong>{}</strong>
            </div>
        </td>'''.format('14', parse_color(cell_value), cell_value if cell_value>0 else '')
    return '''
            <div style=background-color:#BBADA0;width:300px;height:300px>
            <table style=width:100%;height:100%>
              <tr>
                {}
                </tr>
              <tr>
              {}
              </tr>
              <tr>
              {}
              </tr>
              <tr>
              {}
              </tr>
            </table>
            </div>
            '''.format('\n'.join([cell_block(val) for val in layout[0,:]]),
                      '\n'.join([cell_block(val) for val in layout[1,:]]),
                       '\n'.join([cell_block(val) for val in layout[2,:]]),
                       '\n'.join([cell_block(val) for val in layout[3,:]]))

class GameDriver():    
    def __init__(self):
        import numpy as np

    def run_games(self, n, method=lambda layout:np.array([.25,.25,.25,.25]), init_layout=None, early_stop=2048):
        from game import GameLayout
        for i in range(n):
            game = GameLayout()
            if type(init_layout) == np.ndarray:
                game.layout = init_layout
            moves = ['w','a','s','d']
            while game.active:
                # repeatedly attempt to make a move
                for move in self.weighted_shuffle(moves, method(game.layout)):
                    try:
                        game.swipe(move)
                    except:
                        # move didn't work, try next move
                        continue
                    if game.layout.max()>=early_stop:
                        game.active = False
                        break
            # game is over
            self.log_game(game)

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

        except AttributeError:
            self.final_scores = np.array(game.score)
            self.num_moves = np.array(game.num_moves)
            self.layouts = [game.layouts]
            self.final_layouts = [game.final_layout]
            self.moves = [game.moves]
            self.scores = [game.scores]
            self.tile_sums = np.array(game.final_layout.sum())
            self.max_tile = np.array(game.final_layout.max())
