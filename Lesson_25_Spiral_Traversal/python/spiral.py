RIGHT = 0
UP = 1
LEFT = 2
DOWN = 3

class Grid(object):
    def __init__(self, matrix):
        self.matrix = matrix
    
    def _next_position(self, position, direction):
        if direction == RIGHT:
            return (position[0], position[1] + 1)
        elif direction == DOWN:
            return (position[0] + 1, position[1])
        elif direction == LEFT:
            return (position[0], position[1] - 1)
        elif direction == UP:
            return (position[0] - 1, position[1])
        
    def _next_direction(self, direction):
        return {
            RIGHT: DOWN,
            DOWN: LEFT,
            LEFT: UP,
            UP: RIGHT
        }[direction]
        
    def _is_valid_position(self, pos):
        # check if in a valid row
        if (0 <= pos[0] < len(self.matrix)):
            # check if in a a valid column
            if (0 <= pos[1] < len(self.matrix[0])):
                # check we haven't set this position to None yet, which
                # is how we mark if we've been there already
                if (self.matrix[pos[0]][pos[1]] is not None):
                    return True
        return False    
    
    def spiral_print(self):
        # Setup
        remaining = len(self.matrix) * len(self.matrix[0])
        current_direction = RIGHT
        current_position = (0,0)
        result = ''
        # While there are remaining positions to visit
        while remaining > 0:
            remaining -= 1
            
            # to result
            result += str(self.matrix[current_position[0]][current_position[1]]) + ' '
            # set value in matrix at current_position to None
            self.matrix[current_position[0]][current_position[1]] = None
            
            # Get next position
            next_position = self._next_position(current_position,
                                                current_direction)
            # check if it's a valid next position
            if not self._is_valid_position(next_position):
                # Get different direction
                current_direction = self._next_direction(current_direction)
                # Get new position from the direction
                current_position = self._next_position(current_position,
                                                    current_direction)
            # it is valid, so okay to move in direction we already have
            else:
                current_position = self._next_position(current_position,
                                                    current_direction)
            
            
        return result


grid = [[1,  2,  3,  4,  5],
        [6,  7,  8,  9,  10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20]]

print(Grid(grid).spiral_print())