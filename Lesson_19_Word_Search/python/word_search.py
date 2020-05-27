# ASSUMES THAT THE WORD WILL OCCUPY EITHER ENTIRE ROW OR COLUMN
class Grid(object):
    def __init__(self, matrix):
        self.matrix = matrix
        
    def __wordSearchRight(self, index, word):
        # for each letter in this row
        print('STEVEN INDEX: {}'.format(index))
        for i in range(len(self.matrix[index])):
            if word[i] != self.matrix[index][i]:
                return False
        return True
    
    def __wordSearchBottom(self, index, word):
        # for each letter in this column
        for i in range(len(self.matrix)):
            # if we don't get a match we return immediately
            if word[i] != self.matrix[i][index]:
                return False
        return True
    
    def wordSearch(self, word):
        # check all rows
        for i in range(len(self.matrix)):
            # check row i 
            if self.__wordSearchRight(i, word):
                return True
        return True
    
        # check all columns
        for i in range(len(self.matrix[0])):
            # check column i
            if self.__wordSearchBottom(i, word):
                return True
        return False


matrix = [
    ['F', 'A', 'C', 'I'],
    ['O', 'B', 'Q', 'P'],
    ['A', 'N', 'O', 'B'],
    ['M', 'A', 'S', 'S']]

print(Grid(matrix).wordSearch('CQOS'))