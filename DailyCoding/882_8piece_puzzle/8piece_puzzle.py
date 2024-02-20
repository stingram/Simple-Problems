# This problem was asked by Airbnb.

# An 8-puzzle is a game played on a 3 x 3 board of tiles,
# with the ninth tile missing. The remaining tiles are 
# labeled 1 through 8 but shuffled randomly. 
# Tiles may slide horizontally or vertically into an empty space,
# but may not be removed from the board.

# Design a class to represent the board,
# and find a series of steps to bring the
# board to the state [[1, 2, 3], [4, 5, 6], [7, 8, None]].



from collections import deque

class Board:
    def __init__(self, f):
        self.start_board = None
        self.end_goal = [[1,2,3],[4,5,6],[7,8,None]]
        self.f = f
    def _encode_board_state(self,board):
        encoded_board_state = ""
        for i in range(3):
            for j in range(3):
                val = board[i][j]
                if val:
                    encoded_board_state+= str(board[i][j])
                else:
                    encoded_board_state += "0"
        return encoded_board_state

    def _get_options(self, board):
        boards = []
        moves = []
        
        for r in range(3):
            for c in range(3):
                if board[r][c] == None:
                    i = r
                    j = c
        
        deltas_i = [0, -1, 1, 0]
        deltas_j = [-1, 0, 0, 1]

        for delta_i, delta_j in zip(deltas_i,deltas_j):
            new_i = i+delta_i
            new_j = j+delta_j
            if new_i >= 0 and new_i < 3 and new_j >= 0 and new_j < 3:
                new_board = [row[:] for row in board]
                new_board[i][j] = board[new_i][new_j]
                new_board[new_i][new_j] = None
                boards.append(new_board)
                moves.append((delta_i,delta_j))
        return (boards, moves)

    def _helper(self, solution, visited, board, move_mapper):
        q = deque([board])
        visited.add(self._encode_board_state(board))
        while q:
            n = len(q)
            print(f"n: {n}")
            for _ in range(n):
                curr_board = q.popleft()
                print(f"{curr_board}",file=f)
                if curr_board  == self.end_goal:
                    print("DONE")
                    return
                new_boards, moves = self._get_options(curr_board)
                for new_board, move in zip(new_boards, moves):
                    encoded_new_board = self._encode_board_state(new_board)
                    if encoded_new_board not in visited:
                        visited.add(encoded_new_board)
                        
                        # add new board to queue and move to move mapper
                        q.append(new_board)
                        if encoded_new_board not in move_mapper:
                            move_mapper[encoded_new_board] = (self._encode_board_state(curr_board),move)
                        
        return
    
    def _generate_move_list(self, move_mapper):
        print(f"num boards visited: {len(move_mapper)}")
        curr_board = self._encode_board_state(self.end_goal)
        moves = []
        while curr_board != self._encode_board_state(self.start_board):
            curr_board, move = move_mapper[curr_board]
            moves.append(move)
        print(f"num moves: {len(moves)}")
        return moves[::-1]

    def solve(self, board):
        solution = []
        visited = set()
        move_mapper = {}
        self.start_board = board
        self._helper(solution,visited, board, move_mapper)
        move_list = self._generate_move_list(move_mapper)
        return move_list

with open('out.txt', 'w') as f:
    board = [[2,3, None],[1,5,6],[4,7,8]]
    board = [[8,7, 6],[5,4,3],[2,1,None]]
    board_solver = Board(f)
    moves = board_solver.solve(board)
for move in moves:
    print(f"{move}")