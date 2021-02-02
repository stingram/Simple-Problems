#include <vector>
#include <numeric>
#include <algorithm>
#include <unordered_map>
#include <iostream>

class Board
{
    public:
        int N;
        std::unordered_map<int, bool> rows;
        std::unordered_map<int, bool> cols;
        std::unordered_map<int, bool> diag_asc;
        std::unordered_map<int, bool> diag_desc;
        std::vector<int> positions;

    Board(int N)
    {
        positions = std::vector<int>(N,0);
    }
};

bool is_valid(int r, int c, Board& board)
{
    if(board.rows.find(r) == board.rows.end() &&
       board.cols.find(c) == board.cols.end() &&
       board.diag_asc.find(r+c) == board.diag_asc.end() &&
       board.diag_desc.find(r-c) == board.diag_desc.end())
       {
           return true;
       }
       return false;
}


bool n_queens_helper(int row, int N, Board& board)
{
    // base case
    if(row>=N)
    {
        return true;
    }
    // explore this row (search through columns)
    for(int i=0;i<N;i++)
    {
        if(is_valid(row,i, board))
        {
            // make placement
            board.rows[row] = true;
            board.cols[i] = true;
            board.diag_asc[row+i] = true;
            board.diag_desc[row-i] = true;
            board.positions[row] = i;

            // recurse on decision
            if(n_queens_helper(row+1,N,board))
            {
                // we placed queen and all subsequent queens
                return true;
            }

            // else undo placement - backtrack
            board.rows.erase(row);
            board.cols.erase(i);
            board.diag_asc.erase(row+i);
            board.diag_desc.erase(row-i);
            board.positions[row] = 0;
        }
    }

    // got to end and we couldn't place queen, return false
    return false;

}


std::vector<int> n_queens(int N)
{
    Board board = Board(N);
    n_queens_helper(0, N, board);
    return board.positions;
}


int main()
{
    int queens = 4;
    std::vector<int> positions = n_queens(queens);
    for(auto p: positions)
    {
        std::cout << p << ", ";
    }
    std::cout << "\n";
    return 0;
}