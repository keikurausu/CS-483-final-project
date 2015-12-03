
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <stdio.h>

using namespace std;

#define GAME_DIMENSION 6
#define CPU_DEPTH_LIMIT 3
#define CPU_END_LIMIT 8 //when we have this many or less open spaces on the board, just let the CPU go all the way to the end without launching kernels

enum Type { BLUE, GREEN, OPEN };
enum gameMode {AI, HUMAN, DOUBLE_HUMAN};

/*holds data for each block on the gameboard*/
struct block
{
	int value;
	Type team;
};
/*function declarations*/
void setup_game(int x);
void output_game(string filename);
void play_game();
int max_val(block** game_board, Type Max_team, Type Min_team, int depth, int& x, int& y);
int min_val(block** game_board, Type Max_team, Type Min_team, int depth, int& x, int& y);

block** game;  //pointer to array of gameboard blocks
int blue_expanded = 0; //keeps track of total expanded nodes by blue
int green_expanded = 0; //keeps track of total expanded nodes by green
int blue_number_moves = 0;
int green_number_moves = 0;
float average_number_moves;
int blue_score = 0;
int green_score = 0;
double blue_time = 0;
double green_time = 0;
int blocks_occupied = 0; //keeps track of number of blocks which are not OPEN
gameMode game_mode;
/*holds value data for each location of the 5 game boards*/
int gameboard[5][6][6] =
{
	//Keren
	{
		{ 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1 }
	},
	//Narvik
	{
		{ 99, 1, 99, 1, 99, 1 },
		{ 1, 99, 1, 99, 1, 99 },
		{ 99, 1, 99, 1, 99, 1 },
		{ 1, 99, 1, 99, 1, 99 },
		{ 99, 1, 99, 1, 99, 1 },
		{ 1, 99, 1, 99, 1, 99 }
	},
	//Sevastopol
	{
		{ 1, 1, 1, 1, 1, 1 },
		{ 2, 2, 2, 2, 2, 2 },
		{ 4, 4, 4, 4, 4, 4 },
		{ 8, 8, 8, 8, 8, 8 },
		{ 16, 16, 16, 16, 16, 16 },
		{ 32, 32, 32, 32, 32, 32 }
	},
	//Smolensk
	{
		{ 66, 76, 28, 66, 11, 9 },
		{ 31, 39, 50, 8, 33, 14 },
		{ 80, 76, 39, 59, 2, 48 },
		{ 50, 73, 43, 3, 13, 3 },
		{ 99, 45, 72, 87, 49, 4 },
		{ 80, 63, 92, 28, 61, 53 }
	},
	//Westerplatte
	{
		{ 1, 1, 1, 1, 1, 1 },
		{ 1, 3, 4, 4, 3, 1 },
		{ 1, 4, 2, 2, 4, 1 },
		{ 1, 4, 2, 2, 4, 1 },
		{ 1, 3, 4, 4, 3, 1 },
		{ 1, 1, 1, 1, 1, 1 }
	}
};
/*initializes each block in the game*/
void setup_game(int x)
{
	for (int i = 0; i < GAME_DIMENSION; i++)
	{
		for (int j = 0; j < GAME_DIMENSION; j++)
		{
			game[i][j].value = gameboard[x][i][j];
			game[i][j].team = OPEN;
		}
	}
	blue_expanded = 0;
	green_expanded = 0;
	blue_number_moves = 0;
	green_number_moves = 0;
	blue_time = 0;
	green_time = 0;
	blue_score = 0;
	green_score = 0;
	blocks_occupied = 0;
	play_game();
}

/*outputs gameboard result to a file*/
void output_game(string filename)
{
	ofstream outFile(filename.c_str());
	if (outFile.is_open())
	{

		outFile << "Player Blue expanded " << blue_expanded << " nodes" << endl;
		outFile << "Player Green expanded " << green_expanded << " nodes" << endl;
		average_number_moves = float(blue_expanded) / float(blue_number_moves);
		outFile << "Average number of nodes expanded by blue per move: " << average_number_moves << endl;
		average_number_moves = float(green_expanded) / float(green_number_moves);
		outFile << "Average number of nodes expanded by green per move: " << average_number_moves << endl;
		outFile << "Player Blue took " << blue_time << " milliseconds (" << float(blue_time) / float(blue_number_moves) << "ms per move)" << endl;
		outFile << "Player Green took " << green_time << " milliseconds (" << float(green_time) / float(green_number_moves) << "ms per move)" << endl;
		outFile << "Blue total score: " << blue_score << endl;
		outFile << "Green total score: " << green_score << endl;
		if (game_mode != AI)
		{
			if (blue_score > green_score)
				cout << "Blue Wins!" << endl;
			else if (blue_score < green_score)
				cout << "Green Wins!" << endl;
			else
				cout << "Tie!" << endl;
			cout << "Blue final score: " << blue_score << endl;
			cout << "Green final score: " << green_score << endl;
		}
		for (int i = 0; i < GAME_DIMENSION; i++)
		{
			for (int j = 0; j < GAME_DIMENSION; j++)
			{
				if (game[i][j].team == BLUE)
				{
					outFile << 'B';
					if (game_mode!=AI)
						cout << 'B';
				}
				else if (game[i][j].team == GREEN)
				{
					outFile << 'G';
					if (game_mode != AI)
						cout << 'G';
				}
				else
				{
					outFile << '.';
					if (game_mode != AI)
						cout << '.';
				}
			}
			outFile << endl;
			if (game_mode != AI)
				cout << endl;
		}
		outFile.close();
	}
	else
	{
		cout << "error opening " << filename << endl;
	}
}

/*handles actual playing of the game*/
void play_game()
{
	int x, y;
	block** game_copy = new block*[GAME_DIMENSION];
	for (int i = 0; i < GAME_DIMENSION; i++)
	{
		game_copy[i] = new block[GAME_DIMENSION];
	}
	Type current_team = BLUE; //player blue goes first
	Type opponent = GREEN;

	/*take turns going until there are no open spaces left*/
	while (blocks_occupied < GAME_DIMENSION*GAME_DIMENSION)
	{
		int human_x;
		int human_y;
		/*AI is green*/
		if (current_team == GREEN && game_mode == AI || current_team == GREEN && game_mode == HUMAN)
		{
			//make copy before changing things
			for (int i = 0; i < GAME_DIMENSION; i++)
			{
				for (int j = 0; j < GAME_DIMENSION; j++)
				{
					game_copy[i][j].value = game[i][j].value;
					game_copy[i][j].team = game[i][j].team;
				}
			}
			clock_t start = clock();
			max_val(game_copy, current_team, opponent, 1, x, y); //take turn -- once this function returns x and y will hold location of where to go next
			clock_t turn_time = (clock() - start);
			green_time += turn_time;
		}
		/*AI is blue*/
		else if (current_team == BLUE && game_mode == AI)
		{
			//make copy before changing things
			for (int i = 0; i < GAME_DIMENSION; i++)
			{
				for (int j = 0; j < GAME_DIMENSION; j++)
				{
					game_copy[i][j].value = game[i][j].value;
					game_copy[i][j].team = game[i][j].team;
				}
			}
			double start = clock();
			max_val(game_copy, current_team, opponent, 1, x, y); //take turn -- once this function returns x and y will hold location of where to go next
			double turn_time = (clock() - start);
			blue_time += turn_time;
		}
		/*human player's turn*/
		else if (game_mode == HUMAN || game_mode == DOUBLE_HUMAN)
		{
			if (current_team == BLUE)
				cout << "Blue's turn" << endl;
			else
				cout << "Green's turn" << endl;
			cout << "Current Scores:   BLUE: " << blue_score << " GREEN: " << green_score << endl;
			for (int i = 0; i < GAME_DIMENSION; i++)
			{
				for (int j = 0; j < GAME_DIMENSION; j++)
				{
					cout << " ";
					if (game[i][j].team == BLUE)
					{
						cout << "B" << game[i][j].value << " ";
						if (game[i][j].value < 10) //allignment
							cout << " ";
					}
					else if (game[i][j].team == GREEN)
					{
						cout << "G" << game[i][j].value << " ";
						if (game[i][j].value < 10)
							cout << " ";
					}
					else if (game[i][j].team == OPEN)
					{
						cout << "." << game[i][j].value << " ";
						if (game[i][j].value < 10)
							cout << " ";
					}
				}
				cout << endl;
			}
			/*get coordinates from user until valid entry*/
			do{
				cout << endl << "Enter X dimension (1-6)" << endl;
				cin >> human_x;
				cout<< "Enter Y dimension" << endl;
				cin >> human_y;
			} while (game[human_y-1][human_x-1].team != OPEN);
			y = human_y-1;
			x = human_x-1;
		}
		else
		{
			cout << "Error taking turn" << endl;
		}
		//first act as if it is a para drop
		blocks_occupied++;
		game[y][x].team = current_team;
		if (current_team == BLUE)
		{
			blue_score += game[y][x].value;
		}
		else if (current_team == GREEN)
		{
			green_score += game[y][x].value;
		}
		//check for neighbors
		if ((y > 0 && game[y - 1][x].team == current_team) || (y < GAME_DIMENSION - 1 && game[y + 1][x].team == current_team) || (x > 0 && game[y][x - 1].team == current_team) || (x < GAME_DIMENSION - 1 && game[y][x + 1].team == current_team))
		{
			if (y > 0 && game[y - 1][x].team == opponent)
			{
				game[y - 1][x].team = current_team;
				if (current_team == BLUE)
				{
					blue_score += game[y - 1][x].value;
					green_score -= game[y - 1][x].value;
				}
				else
				{
					green_score += game[y - 1][x].value;
					blue_score -= game[y - 1][x].value;
				}

			}
			if (y < GAME_DIMENSION - 1 && game[y + 1][x].team == opponent)
			{
				game[y + 1][x].team = current_team;
				if (current_team == BLUE)
				{
					blue_score += game[y + 1][x].value;
					green_score -= game[y + 1][x].value;
				}
				else
				{
					green_score += game[y + 1][x].value;
					blue_score -= game[y + 1][x].value;
				}
			}
			if (x > 0 && game[y][x - 1].team == opponent)
			{
				game[y][x - 1].team = current_team;
				if (current_team == BLUE)
				{
					blue_score += game[y][x - 1].value;
					green_score -= game[y][x - 1].value;
				}
				else
				{
					green_score += game[y][x - 1].value;
					blue_score -= game[y][x - 1].value;
				}
			}
			if (x < GAME_DIMENSION - 1 && game[y][x + 1].team == opponent)
			{
				game[y][x + 1].team = current_team;
				if (current_team == BLUE)
				{
					blue_score += game[y][x + 1].value;
					green_score -= game[y][x + 1].value;
				}
				else
				{
					green_score += game[y][x + 1].value;
					blue_score -= game[y][x + 1].value;
				}
			}
		}
		if (current_team == BLUE)
		{
			current_team = GREEN;
			opponent = BLUE;
			blue_number_moves += 1;
		}
		else
		{
			current_team = BLUE;
			opponent = GREEN;
			green_number_moves += 1;
		}
	}
	//memory cleanup
	for (int i = 0; i < GAME_DIMENSION; i++)
	{
		delete[] game_copy[i];
	}
	delete[] game_copy;

}

int max_val(block** game_board, Type Max_team, Type Min_team, int depth, int& x, int& y)
{
	int best = -1000; //best value so far is held here
	int best_evaluation = -1000; //used for evaluation function

	for (int i = 0; i < GAME_DIMENSION; i++)
	{
		for (int j = 0; j < GAME_DIMENSION; j++)
		{
			//check if we have reached a terminal node
			if (blocks_occupied + depth - 1 == GAME_DIMENSION*GAME_DIMENSION - 1)
			{
				if (game_board[i][j].team == OPEN)
				{
					// This open node will be expanded.
					if (Max_team == BLUE) {
						blue_expanded += 1;
					}
					else if (Max_team == GREEN) {
						green_expanded += 1;
					}

					//set location values which will be sent back
					x = j;
					y = i;
					int utility = game_board[i][j].value;
					//check for neighbors
					if ((i > 0 && game_board[i - 1][j].team == Max_team) || (i < GAME_DIMENSION - 1 && game_board[i + 1][j].team == Max_team) || (j > 0 && game_board[i][j - 1].team == Max_team) || (j < GAME_DIMENSION - 1 && game_board[i][j + 1].team == Max_team))
					{
						if (i > 0 && game_board[i - 1][j].team == Min_team)
						{
							utility += game_board[i - 1][j].value*2;
						}
						if (i < GAME_DIMENSION - 1 && game_board[i + 1][j].team == Min_team)
						{
							utility += game_board[i + 1][j].value*2;
						}
						if (j > 0 && game_board[i][j - 1].team == Min_team)
						{
							utility += game_board[i][j - 1].value*2;
						}
						if (j < GAME_DIMENSION - 1 && game_board[i][j + 1].team == Min_team)
						{
							utility += game_board[i][j + 1].value*2;
						}
					}
					return utility;
				}
			}
			//not at terminal node and depth limit not reached
			else if (depth < CPU_DEPTH_LIMIT || GAME_DIMENSION*GAME_DIMENSION - blocks_occupied < CPU_END_LIMIT)
			{
				int local_best;
				int x_loc; //used to hold return value
				int y_loc;
				if (game_board[i][j].team == OPEN)
				{
					// This open node will be expanded.
					if (Max_team == BLUE) {
						blue_expanded += 1;
					}
					else if (Max_team == GREEN) {
						green_expanded += 1;
					}

					//MAKE COPY EACH TIME
					block** copy = new block*[GAME_DIMENSION];
					for (int k = 0; k < GAME_DIMENSION; k++)
					{
						copy[k] = new block[GAME_DIMENSION];
					}
					for (int k = 0; k < GAME_DIMENSION; k++)
					{
						for (int m = 0; m < GAME_DIMENSION; m++)
						{
							copy[k][m].value = game_board[k][m].value;
							copy[k][m].team = game_board[k][m].team;
						}
					}
					//perform para drop
					copy[i][j].team = Max_team;
					//check for neighbors
					if ((i > 0 && copy[i - 1][j].team == Max_team) || (i < GAME_DIMENSION - 1 && copy[i + 1][j].team == Max_team) || (j > 0 && copy[i][j - 1].team == Max_team) || (j < GAME_DIMENSION - 1 && copy[i][j + 1].team == Max_team))
					{
						if (i > 0 && copy[i - 1][j].team == Min_team)
						{
							copy[i - 1][j].team = Max_team;
						}
						if (i < GAME_DIMENSION - 1 && copy[i + 1][j].team == Min_team)
						{
							copy[i + 1][j].team = Max_team;
						}
						if (j > 0 && copy[i][j - 1].team == Min_team)
						{
							copy[i][j - 1].team = Max_team;
						}
						if (j < GAME_DIMENSION - 1 && copy[i][j + 1].team == Min_team)
						{
							copy[i][j + 1].team = Max_team;
						}
					}
					local_best = min_val(copy, Max_team, Min_team, depth + 1, x_loc, y_loc);

					//update best location
					if (local_best > best)
					{
						best = local_best;
						x = x_loc;
						y = y_loc;
					}
					//memory cleanup
					for (int k = 0; k < GAME_DIMENSION; k++)
					{
						delete[] copy[k];
					}
					delete[] copy;
				}
			}
			/*perform evaluation function since depth limit reached*/
			else if (depth == CPU_DEPTH_LIMIT)
			{
				int max_total = 0;
				int min_total = 0;
				if (game_board[i][j].team == OPEN)
				{
					// This open node will be expanded.
					if (Max_team == BLUE) {
						blue_expanded += 1;
					}
					else if (Max_team == GREEN) {
						green_expanded += 1;
					}

					//MAKE COPY EACH TIME
					int local_best;
					block** copy = new block*[GAME_DIMENSION];
					for (int k = 0; k < GAME_DIMENSION; k++)
					{
						copy[k] = new block[GAME_DIMENSION];
					}
					for (int k = 0; k < GAME_DIMENSION; k++)
					{
						for (int m = 0; m < GAME_DIMENSION; m++)
						{
							copy[k][m].value = game_board[k][m].value;
							copy[k][m].team = game_board[k][m].team;
						}
					}
					//perform para drop
					copy[i][j].team = Max_team;
					//check for neighbors
					if ((i > 0 && copy[i - 1][j].team == Max_team) || (i < GAME_DIMENSION - 1 && copy[i + 1][j].team == Max_team) || (j > 0 && copy[i][j - 1].team == Max_team) || (j < GAME_DIMENSION - 1 && copy[i][j + 1].team == Max_team))
					{
						if (i > 0 && copy[i - 1][j].team == Min_team)
						{
							copy[i - 1][j].team = Max_team;
						}
						if (i < GAME_DIMENSION - 1 && copy[i + 1][j].team == Min_team)
						{
							copy[i + 1][j].team = Max_team;
						}
						if (j > 0 && copy[i][j - 1].team == Min_team)
						{
							copy[i][j - 1].team = Max_team;
						}
						if (j < GAME_DIMENSION - 1 && copy[i][j + 1].team == Min_team)
						{
							copy[i][j + 1].team = Max_team;
						}
					}

					/*add up all current values on board of max_team and min_team and compute the difference*/
					for (int k = 0; k < GAME_DIMENSION; k++)
					{
						for (int m = 0; m < GAME_DIMENSION; m++)
						{
							if (copy[k][m].team == Max_team)
							{
								max_total += copy[k][m].value;
							}
							else if (copy[k][m].team == Min_team)
							{
								min_total += copy[k][m].value;
							}
						}
					}
					local_best = max_total - min_total;
					if (local_best > best_evaluation)
					{
						best_evaluation = local_best;
						x = j;
						y = i;
					}
					//memory cleanup
					for (int k = 0; k < GAME_DIMENSION; k++)
					{
						delete[] copy[k];
					}
					delete[] copy;
				}
			}
		}
	}
	if (depth < CPU_DEPTH_LIMIT || GAME_DIMENSION*GAME_DIMENSION - blocks_occupied < CPU_END_LIMIT)
	{
		return best;
	}
	else
	{
		return best_evaluation;
	}

}

int min_val(block** game_board, Type Max_team, Type Min_team, int depth, int& x, int& y)
{
	int best = 1000; //best value (in this case lowest value) so far is held here
	int best_evaluation = 1000; //used for evaluation function

	for (int i = 0; i < GAME_DIMENSION; i++)
	{
		for (int j = 0; j < GAME_DIMENSION; j++)
		{
			//check if we have reached a terminal node
			if (blocks_occupied + depth - 1 == GAME_DIMENSION*GAME_DIMENSION - 1)
			{
				if (game_board[i][j].team == OPEN)
				{
					// This open node will be expanded.
					if (Max_team == BLUE) {
						blue_expanded += 1;
					}
					else if (Max_team == GREEN) {
						green_expanded += 1;
					}

					//set location values which will be sent back
					x = j;
					y = i;
					int utility = game_board[i][j].value;
					//check for neighbors
					if ((i > 0 && game_board[i - 1][j].team == Min_team) || (i < GAME_DIMENSION - 1 && game_board[i + 1][j].team == Min_team) || (j > 0 && game_board[i][j - 1].team == Min_team) || (j < GAME_DIMENSION - 1 && game_board[i][j + 1].team == Min_team))
					{
						if (i > 0 && game_board[i - 1][j].team == Max_team)
						{
							utility += game_board[i - 1][j].value*2;
						}
						if (i < GAME_DIMENSION - 1 && game_board[i + 1][j].team == Max_team)
						{
							utility += game_board[i + 1][j].value*2;
						}
						if (j > 0 && game_board[i][j - 1].team == Max_team)
						{
							utility += game_board[i][j - 1].value*2;
						}
						if (j < GAME_DIMENSION - 1 && game_board[i][j + 1].team == Max_team)
						{
							utility += game_board[i][j + 1].value*2;
						}
					}
					return utility;
				}
			}
			//not at terminal node and depth limit not reached
			else if (depth < CPU_DEPTH_LIMIT || GAME_DIMENSION*GAME_DIMENSION - blocks_occupied < CPU_END_LIMIT)
			{
				int local_best;
				int x_loc; //used to hold return value
				int y_loc;
				if (game_board[i][j].team == OPEN)
				{
					// This open node will be expanded.
					if (Max_team == BLUE) {
						blue_expanded += 1;
					}
					else if (Max_team == GREEN) {
						green_expanded += 1;
					}

					//MAKE COPY EACH TIME
					block** copy = new block*[GAME_DIMENSION];
					for (int k = 0; k < GAME_DIMENSION; k++)
					{
						copy[k] = new block[GAME_DIMENSION];
					}
					for (int k = 0; k < GAME_DIMENSION; k++)
					{
						for (int m = 0; m < GAME_DIMENSION; m++)
						{
							copy[k][m].value = game_board[k][m].value;
							copy[k][m].team = game_board[k][m].team;
						}
					}
					//perform para drop
					copy[i][j].team = Min_team;
					//check for neighbors
					if ((i > 0 && copy[i - 1][j].team == Min_team) || (i < GAME_DIMENSION - 1 && copy[i + 1][j].team == Min_team) || (j > 0 && copy[i][j - 1].team == Min_team) || (j < GAME_DIMENSION - 1 && copy[i][j + 1].team == Min_team))
					{
						if (i > 0 && copy[i - 1][j].team == Max_team)
						{
							copy[i - 1][j].team = Min_team;
						}
						if (i < GAME_DIMENSION - 1 && copy[i + 1][j].team == Max_team)
						{
							copy[i + 1][j].team = Min_team;
						}
						if (j > 0 && copy[i][j - 1].team == Max_team)
						{
							copy[i][j - 1].team = Min_team;
						}
						if (j < GAME_DIMENSION - 1 && copy[i][j + 1].team == Max_team)
						{
							copy[i][j + 1].team = Min_team;
						}
					}
					local_best = max_val(copy, Max_team, Min_team, depth + 1, x_loc, y_loc);

					//update best location if necessary
					if (local_best < best)
					{
						best = local_best;
						x = x_loc;
						y = y_loc;
					}
					//memory cleanup
					for (int k = 0; k < GAME_DIMENSION; k++)
					{
						delete[] copy[k];
					}
					delete[] copy;
				}
			}
			/*perform evaluation function since depth limit reached*/
			else if (depth == CPU_DEPTH_LIMIT)
			{
				int max_total = 0;
				int min_total = 0;
				if (game_board[i][j].team == OPEN)
				{
					// This open node will be expanded.
					if (Max_team == BLUE) {
						blue_expanded += 1;
					}
					else if (Max_team == GREEN) {
						green_expanded += 1;
					}

					//MAKE COPY EACH TIME
					int local_best;
					block** copy = new block*[GAME_DIMENSION];
					for (int k = 0; k < GAME_DIMENSION; k++)
					{
						copy[k] = new block[GAME_DIMENSION];
					}
					for (int k = 0; k < GAME_DIMENSION; k++)
					{
						for (int m = 0; m < GAME_DIMENSION; m++)
						{
							copy[k][m].value = game_board[k][m].value;
							copy[k][m].team = game_board[k][m].team;
						}
					}
					//perform para drop
					copy[i][j].team = Min_team;
					//check for neighbors
					if ((i > 0 && copy[i - 1][j].team == Min_team) || (i < GAME_DIMENSION - 1 && copy[i + 1][j].team == Min_team) || (j > 0 && copy[i][j - 1].team == Min_team) || (j < GAME_DIMENSION - 1 && copy[i][j + 1].team == Min_team))
					{
						if (i > 0 && copy[i - 1][j].team == Max_team)
						{
							copy[i - 1][j].team = Min_team;
						}
						if (i < GAME_DIMENSION - 1 && copy[i + 1][j].team == Max_team)
						{
							copy[i + 1][j].team = Min_team;
						}
						if (j > 0 && copy[i][j - 1].team == Max_team)
						{
							copy[i][j - 1].team = Min_team;
						}
						if (j < GAME_DIMENSION - 1 && copy[i][j + 1].team == Max_team)
						{
							copy[i][j + 1].team = Min_team;
						}
					}

					/*add up all current values on board of max_team and min_team and compute the difference*/
					for (int k = 0; k < GAME_DIMENSION; k++)
					{
						for (int m = 0; m < GAME_DIMENSION; m++)
						{
							if (copy[k][m].team == Max_team)
							{
								max_total += copy[k][m].value;
							}
							else if (copy[k][m].team == Min_team)
							{
								min_total += copy[k][m].value;
							}
						}
					}
					/*we want to minimize the difference*/
					local_best = max_total - min_total;
					if (local_best < best_evaluation)
					{
						best_evaluation = local_best;
						x = j;
						y = i;
					}
					//memory cleanup
					for (int k = 0; k < GAME_DIMENSION; k++)
					{
						delete[] copy[k];
					}
					delete[] copy;
				}
			}
		}
	}
	if (depth < CPU_DEPTH_LIMIT || GAME_DIMENSION*GAME_DIMENSION - blocks_occupied < CPU_END_LIMIT)
	{
		return best;
	}
	else
	{
		return best_evaluation;
	}
}

int main()
{
	double endTime;
	double programTime;
	int mode;
	int mapNum = 0;
	do{
		cout << "Welcome to War Game! Enter game mode (0 for AI vs AI, 1 for AI vs Human, 2 for Human vs Human)" << endl;
		cin >> mode;
	} while (mode != 0 && mode != 1 && mode != 2);
	double startTime = clock(); //start keeping track of time of execution
	if (mode != 0)
	{
		do{
			cout << "Enter map number to play (0 - 4)" << endl;
			cin >> mapNum;
		} while (mapNum > 4 || mapNum < 0);
	}
	if (mode == 0){
		game_mode = AI;
	}
	else if(mode == 1){
		game_mode = HUMAN;
	}
	else{
		game_mode = DOUBLE_HUMAN;
	}

	game = new block*[GAME_DIMENSION]; //pointer to gameboard
	for (int i = 0; i < GAME_DIMENSION; i++)
	{
		game[i] = new block[GAME_DIMENSION];
	}

	/*human play setup*/
	if (mode != 0)
	{
		/*play game using selected map*/
		setup_game(mapNum);
		switch (mapNum)
		{
		case 0:
			output_game("Keren_output.txt");
		case 1:
			output_game("Narvik_output.txt");
		case 2:
			output_game("Sevastopol_output.txt");
		case 3:
			output_game("Smolesnk_output.txt");
		case 4:
			output_game("Westerplatte_output.txt");
		default:
			output_game("Keren_output.txt");
		}
	}
	/*AI vs AI setup*/
	else
	{
		//play game using Keren map
		setup_game(0);
		output_game("Keren_output.txt");

		//play game using Narvik map
		setup_game(1);
		output_game("Narvik_output.txt");

		//play game using Sevastopol map
		setup_game(2);
		output_game("Sevastopol_output.txt");

		//play game using Smolensk map
		setup_game(3);
		output_game("Smolesnk_output.txt");

		//play game using Westerplatte map
		setup_game(4);
		output_game("Westerplatte_output.txt");
	}
	
	//memory cleanup
	for (int i = 0; i < GAME_DIMENSION; i++)
	{
		delete[] game[i];
	}
	delete[] game;
	endTime = clock();
	programTime = (endTime - startTime) / 1000;
	cout << "Process Execution Time " << programTime << "s" << endl;
	cin.ignore();
	cin.get(); // pause program to view results
	return 0;
}

//CUDA reference code
/*
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
int i = threadIdx.x;
c[i] = a[i] + b[i];
}

int main()
{
const int arraySize = 5;
const int a[arraySize] = { 1, 2, 3, 4, 5 };
const int b[arraySize] = { 10, 20, 30, 40, 50 };
int c[arraySize] = { 0 };

// Add vectors in parallel.
cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "addWithCuda failed!");
return 1;
}
printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
c[0], c[1], c[2], c[3], c[4]);

// cudaDeviceReset must be called before exiting in order for profiling and
// tracing tools such as Nsight and Visual Profiler to show complete traces.
cudaStatus = cudaDeviceReset();
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaDeviceReset failed!");
return 1;
}
cin.get();
return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
int *dev_a = 0;
int *dev_b = 0;
int *dev_c = 0;
cudaError_t cudaStatus;

// Choose which GPU to run on, change this on a multi-GPU system.
cudaStatus = cudaSetDevice(0);
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
goto Error;
}

// Allocate GPU buffers for three vectors (two input, one output)    .
cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaMalloc failed!");
goto Error;
}

cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaMalloc failed!");
goto Error;
}

cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaMalloc failed!");
goto Error;
}

// Copy input vectors from host memory to GPU buffers.
cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaMemcpy failed!");
goto Error;
}

cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaMemcpy failed!");
goto Error;
}

// Launch a kernel on the GPU with one thread for each element.
addKernel<<< 1, size >>>(dev_c, dev_a, dev_b);

// Check for any errors launching the kernel
cudaStatus = cudaGetLastError();
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
goto Error;
}

// cudaDeviceSynchronize waits for the kernel to finish, and returns
// any errors encountered during the launch.
cudaStatus = cudaDeviceSynchronize();
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
goto Error;
}

// Copy output vector from GPU buffer to host memory.
cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaMemcpy failed!");
goto Error;
}

Error:
cudaFree(dev_c);
cudaFree(dev_a);
cudaFree(dev_b);

return cudaStatus;
}
*/