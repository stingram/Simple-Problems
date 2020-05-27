// spiral.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>




class Grid
{
private:
	std::vector<std::vector<int>> grid;
	enum class DIRECTIONS { RIGHT, UP, LEFT, DOWN };

	std::pair<int, int> _next_position(const std::pair<int, int>& position,
		DIRECTIONS direction)
	{
		if (direction == DIRECTIONS::RIGHT)
			return std::pair<int, int>(position.first, position.second + 1);
		else if (direction == DIRECTIONS::DOWN)
			return std::pair<int, int>(position.first + 1, position.second);
		else if (direction == DIRECTIONS::LEFT)
			return std::pair<int, int>(position.first, position.second - 1);
		else if (direction == DIRECTIONS::UP)
			return std::pair<int, int>(position.first - 1, position.second);
	}


	DIRECTIONS next_direction(DIRECTIONS direction)
	{
		if (direction == DIRECTIONS::RIGHT)
			return DIRECTIONS::DOWN;
		else if (direction == DIRECTIONS::DOWN)
			return DIRECTIONS::LEFT;
		else if (direction == DIRECTIONS::LEFT)
			return DIRECTIONS::UP;
		else if (direction == DIRECTIONS::UP)
			return DIRECTIONS::RIGHT;
	}

	bool is_valid_position(std::pair<int, int> pos)
	{
		if (0 <= pos.first && pos.first < grid.size())
		{
			if (0 <= pos.second && pos.second < grid[0].size())
			{
				if (grid[pos.first][pos.second] != NULL)
				{
					return true;
				}
			}
		}
		return false;
	}



public:
	Grid(std::vector<std::vector<int>> grid)
	{
		this->grid = grid;
	}

	void spiral_print()
	{
		int remaining = grid.size() * grid[0].size();
		DIRECTIONS current_direction = DIRECTIONS::RIGHT;
		std::pair<int, int> current_position = { 0,0 };
		std::string result;
		while (remaining > 0)
		{
			remaining -= 1;
			result += std::to_string(grid[current_position.first]
				[current_position.second]) + " ";
			grid[current_position.first][current_position.second] = NULL;

			std::pair<int, int> next_position = _next_position(current_position,
				current_direction);
			if (!is_valid_position(next_position))
			{
				current_direction = next_direction(current_direction);
				current_position = _next_position(
					current_position, current_direction);
			}
			else
			{
				current_position = _next_position(
					current_position, current_direction);
			}
		}
		std::cout << "RESULT: " << result << "\n";
	}
};


int main()
{
	std::vector<std::vector<int>> grid = { {1,  2,  3,  4,  5},
										  {6,  7,  8,  9,  10},
										  {11, 12, 13, 14, 15},
										  {16, 17, 18, 19, 20} };

	Grid(grid).spiral_print();
	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
