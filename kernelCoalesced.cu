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
#define CPU_END_LIMIT 8 //when we have this many or less open spaces on the board, just let the CPU go all the way to the end without launching kernels since we don't have many nodes
#define THREADS 256

/*function declarations*/
void setup_game(int x);
void output_game(string filename);
void play_game();
int max_val(char** game_board, char Max_team, char Min_team, int depth, int& x, int& y);
int min_val(char** game_board, char Max_team, char Min_team, int depth, int& x, int& y);
cudaError_t searchHelper(char* input, int size, int nodes, char Max_team, char Min_team);

/*global variable declarations*/
enum gameMode { AI, HUMAN, DOUBLE_HUMAN }; //AI for AIvsAI, HUMAN for human vs AI, DOUBLE_HUMAN for human vs human

__constant__ int Vc[GAME_DIMENSION][GAME_DIMENSION]; //constant memory which holds gameboard values for device computations

int values[GAME_DIMENSION][GAME_DIMENSION]; //constant array of gameboard values which is filled based on selected map. Used by CPU
char** game;  //pointer to our gameboard. 'o' for open, 'b' for blue, 'g' for green team occupying a space
char* hostArray; // array which holds stuff that needs to be copied to device
int* hostOutput; //pointer to host output array
int arrayCount; //keeps track of where we are in the depth first search
int kernelLaunchCount = 0; //keeps track of number of kernels launched for debugging purposes
int pass = 1; //pass 1 fill array which is sent to kernel, pass 2 read from array filled by kernel
int pMode = 1; //used to compare performance between serial and parallel version. Set to 0 and set CPU_DEPTH_LIMIT to 4 to disable parallel code
int blue_score = 0;
int green_score = 0;
double blue_time = 0; //time taken by blue AI. Used for computing average time per turn
double green_time = 0; //time taken by green AI.
int currentSize = 0; //holds current number of leaf nodes.
int blocks_occupied = 0; //keeps track of number of blocks which are not OPEN
gameMode game_mode; //current gamemode

/*holds value data for the 5 game boards*/
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

/* input holds game states up to depth 3, output will hold 3 values to be evaluated at each depth 3 node as follows. [utility value][y-coord][x-coord] */
__global__ void gameSearchKernel(char* input, int* output, const int size, int nodes, char Max_team, char Min_team)
{
	char intermediate_buffer[GAME_DIMENSION][GAME_DIMENSION];
	int max_total = 0;
	int min_total = 0;
	int best_evaluation = 10000; //holds best value found so far
	int x; //holds best x coordinate
	int y; //holds best y coordinate
	//int index = blockIdx.x*blockDim.x*GAME_DIMENSION*GAME_DIMENSION + threadIdx.x*GAME_DIMENSION*GAME_DIMENSION; //extract index helper
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < size)
	{
		for (int i = 0; i < GAME_DIMENSION; i++)
		{
			for (int j = 0; j < GAME_DIMENSION; j++)
			{
				//intermediate_buffer[i][j] = input[index + i*GAME_DIMENSION + j]; //extract the root this thread needs to explore
				intermediate_buffer[i][j] = input[size*(i*GAME_DIMENSION + j) + index];
			}
		}
		for (int i = 0; i < GAME_DIMENSION; i++)
		{
			for (int j = 0; j < GAME_DIMENSION; j++)
			{
				if (intermediate_buffer[i][j] == 'o')
				{
					max_total = 0;
					min_total = 0;
					int local_best;
					//MAKE COPY EACH TIME
					char copy[GAME_DIMENSION][GAME_DIMENSION];
					for (int k = 0; k < GAME_DIMENSION; k++)
					{
						for (int m = 0; m < GAME_DIMENSION; m++)
						{
							copy[k][m] = intermediate_buffer[k][m];
						}
					}
					//perform para drop
					copy[i][j] = Min_team;
					//check for neighbors
					if ((i > 0 && copy[i - 1][j] == Min_team) || (i < GAME_DIMENSION - 1 && copy[i + 1][j] == Min_team) || (j > 0 && copy[i][j - 1] == Min_team) || (j < GAME_DIMENSION - 1 && copy[i][j + 1] == Min_team))
					{
						if (i > 0 && copy[i - 1][j] == Max_team)
						{
							copy[i - 1][j] = Min_team;
						}
						if (i < GAME_DIMENSION - 1 && copy[i + 1][j] == Max_team)
						{
							copy[i + 1][j] = Min_team;
						}
						if (j > 0 && copy[i][j - 1] == Max_team)
						{
							copy[i][j - 1] = Min_team;
						}
						if (j < GAME_DIMENSION - 1 && copy[i][j + 1] == Max_team)
						{
							copy[i][j + 1] = Min_team;
						}
					}

					/*add up all current values on board of max_team and min_team and compute the difference*/
					for (int k = 0; k < GAME_DIMENSION; k++)
					{
						for (int m = 0; m < GAME_DIMENSION; m++)
						{
							if (copy[k][m] == Max_team)
							{
								max_total += Vc[k][m];
							}
							else if (copy[k][m] == Min_team)
							{
								min_total += Vc[k][m];
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
				}
			}
		}
		//write to output
		index = blockIdx.x*blockDim.x * 3 + threadIdx.x * 3;
		output[index] = best_evaluation;
		output[index + 1] = y;
		output[index + 2] = x;
	}
}

/* Helper function for launching GPU. Gets everything ready, then launches kernel, and copies memory back to host. //size is number of game boards */
cudaError_t searchHelper(char* input, int size, int nodes, char Max_team, char Min_team)
{
	char* dev_input = 0;
	int* dev_output = 0;
	cudaError_t cudaStatus;
	
	//allocate host memory
	hostOutput = (int*)malloc(3 * size * sizeof(int));
	
	// Allocate GPU memory for output array
	cudaStatus = cudaMalloc((void**)&dev_output, 3 * size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc device output failed!");
		goto Error;
	}

	// Allocate GPU memory for input array
	cudaStatus = cudaMalloc((void**)&dev_input, GAME_DIMENSION * GAME_DIMENSION * size * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc device input failed!");
		goto Error;
	}
	
	// Copy input from host memory to GPU buffer.
	cudaStatus = cudaMemcpy(dev_input, input, GAME_DIMENSION * GAME_DIMENSION * size * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy 1 host to device failed!");
		goto Error;
	}

	// Launch kernel!
	gameSearchKernel << < (size - 1) / THREADS + 1, THREADS >> >(dev_input, dev_output, size, nodes, Max_team, Min_team);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching gameSearchKernel!\n", cudaStatus);
		goto Error;
	}
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("gameSearchKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// Copy output from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(hostOutput, dev_output, 3* size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy 2 device to host failed!");
		goto Error;
	}
	
	//free device memory
	cudaStatus = cudaFree(dev_input);
	if (cudaStatus != cudaSuccess) {
		printf("cudaFree input failed!");
		goto Error;
	}
	cudaStatus = cudaFree(dev_output);
	if (cudaStatus != cudaSuccess) {
		printf("cudaFree output failed!");
		goto Error;
	}
	return cudaStatus;

Error:
	//free device memory if we get an error
	cudaFree(dev_input);
	cudaFree(dev_output);
	return cudaStatus;
}

/*handles actual playing of the game*/
void play_game()
{
	int x, y;
	char** game_copy = new char*[GAME_DIMENSION];
	for (int i = 0; i < GAME_DIMENSION; i++)
	{
		game_copy[i] = new char[GAME_DIMENSION];
	}
	char current_team = 'b'; //player 'b' goes first
	char opponent = 'g';

	/*take turns going until there are no open spaces left*/
	while (blocks_occupied < GAME_DIMENSION*GAME_DIMENSION)
	{
		int human_x;
		int human_y;
		/*AI is green*/
		if (current_team == 'g' && game_mode == AI || current_team == 'g' && game_mode == HUMAN)
		{
			//make copy before changing things
			for (int i = 0; i < GAME_DIMENSION; i++)
			{
				for (int j = 0; j < GAME_DIMENSION; j++)
				{
					game_copy[i][j] = game[i][j];
				}
			}
			/*check if we need to launch GPU*/
			if (GAME_DIMENSION*GAME_DIMENSION - blocks_occupied >= CPU_END_LIMIT && pMode == 1)
			{
				kernelLaunchCount++;
				currentSize = (GAME_DIMENSION*GAME_DIMENSION - blocks_occupied)*(GAME_DIMENSION*GAME_DIMENSION - blocks_occupied - 1)*(GAME_DIMENSION*GAME_DIMENSION - blocks_occupied - 2); //get current number of leaf nodes
				hostArray = new char[currentSize*GAME_DIMENSION*GAME_DIMENSION]; //allocate memory to store data that needs to be copied to device
				arrayCount = 0;
				cudaError_t cudaStatus;
				double start = clock();
				/*do first pass which launches the kernel to compute the depth 4 values*/
				pass = 1;
				max_val(game_copy, current_team, opponent, 1, x, y); //take turn -- once this function returns x and y will hold location of where to go next
				//cout << endl << "pass 1 complete" << endl;
				//cin.get();
				cudaStatus = searchHelper(hostArray, currentSize, GAME_DIMENSION*GAME_DIMENSION - blocks_occupied - 2, current_team, opponent);
				if (cudaStatus != cudaSuccess) {
					printf("searchHelper failed!");
				}
				delete[] hostArray; // free host input array

				/*do second pass which reads values from array filled by kernel*/
				pass = 2;
				arrayCount = 0;
				max_val(game_copy, current_team, opponent, 1, x, y);
				//cout << endl << arrayCount << endl;
				double turn_time = (clock() - start);
				green_time += turn_time;
				
				//free host memory
				free(hostOutput);
			}
			else{
				double start = clock();
				max_val(game_copy, current_team, opponent, 1, x, y); //take turn -- once this function returns x and y will hold location of where to go next
				double turn_time = (clock() - start);
				green_time += turn_time;
			}
		}
		/*AI is blue*/
		else if (current_team == 'b' && game_mode == AI)
		{
			//make copy before changing things
			for (int i = 0; i < GAME_DIMENSION; i++)
			{
				for (int j = 0; j < GAME_DIMENSION; j++)
				{
					game_copy[i][j] = game[i][j];
				}
			}
			/*check if we need to launch GPU*/
			if (GAME_DIMENSION*GAME_DIMENSION - blocks_occupied >= CPU_END_LIMIT && pMode == 1)
			{
				kernelLaunchCount++;
				currentSize = (GAME_DIMENSION*GAME_DIMENSION - blocks_occupied)*(GAME_DIMENSION*GAME_DIMENSION - blocks_occupied - 1)*(GAME_DIMENSION*GAME_DIMENSION - blocks_occupied - 2); //get current number of leaf nodes
				hostArray = new char[currentSize*GAME_DIMENSION*GAME_DIMENSION]; //allocate memory to store data that needs to be copied to device
				arrayCount = 0;
				cudaError_t cudaStatus;
				double start = clock();
				/*do first pass which launches the kernel to compute the depth 4 values*/
				pass = 1;
				max_val(game_copy, current_team, opponent, 1, x, y); //take turn -- once this function returns x and y will hold location of where to go next
				cudaStatus = searchHelper(hostArray, currentSize, GAME_DIMENSION*GAME_DIMENSION - blocks_occupied - 2, current_team, opponent);
				if (cudaStatus != cudaSuccess) {
					printf("searchHelper failed!");
				}
				delete[] hostArray; // free host input array

				/*do second pass which reads values from array filled by kernel*/
				pass = 2;
				arrayCount = 0;
				max_val(game_copy, current_team, opponent, 1, x, y);
				double turn_time = (clock() - start);
				blue_time += turn_time;
				free(hostOutput);

			}
			else{
				double start = clock();
				max_val(game_copy, current_team, opponent, 1, x, y); //take turn -- once this function returns x and y will hold location of where to go next
				double turn_time = (clock() - start);
				blue_time += turn_time;
			}
		}
		/*human player's turn*/
		else if (game_mode == HUMAN || game_mode == DOUBLE_HUMAN)
		{
			if (current_team == 'b')
				cout << "Blue's turn" << endl;
			else
				cout << "Green's turn" << endl;
			cout << "Current Scores:   BLUE: " << blue_score << " GREEN: " << green_score << endl;
			for (int i = 0; i < GAME_DIMENSION; i++)
			{
				for (int j = 0; j < GAME_DIMENSION; j++)
				{
					cout << " ";
					if (game[i][j] == 'b')
					{
						cout << "B" << values[i][j] << " ";
						if (values[i][j] < 10) //allignment
							cout << " ";
					}
					else if (game[i][j] == 'g')
					{
						cout << "G" << values[i][j] << " ";
						if (values[i][j] < 10)
							cout << " ";
					}
					else if (game[i][j] == 'o')
					{
						cout << "." << values[i][j] << " ";
						if (values[i][j] < 10)
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
			} while (game[human_y-1][human_x-1] != 'o' && human_y < 7 && human_y > 0 && human_x < 7 && human_x > 0);
			y = human_y-1;
			x = human_x-1;
		}
		else
		{
			cout << "Error taking turn" << endl;
		}
		//first act as if it is a para drop
		blocks_occupied++;
		game[y][x] = current_team;
		if (current_team == 'b')
		{
			blue_score += values[y][x];
		}
		else if (current_team == 'g')
		{
			green_score += values[y][x];
		}
		//check for neighbors
		if ((y > 0 && game[y - 1][x] == current_team) || (y < GAME_DIMENSION - 1 && game[y + 1][x] == current_team) || (x > 0 && game[y][x - 1] == current_team) || (x < GAME_DIMENSION - 1 && game[y][x + 1] == current_team))
		{
			if (y > 0 && game[y - 1][x] == opponent)
			{
				game[y - 1][x] = current_team;
				if (current_team == 'b')
				{
					blue_score += values[y - 1][x];
					green_score -= values[y - 1][x];
				}
				else
				{
					green_score += values[y - 1][x];
					blue_score -= values[y - 1][x];
				}

			}
			if (y < GAME_DIMENSION - 1 && game[y + 1][x] == opponent)
			{
				game[y + 1][x] = current_team;
				if (current_team == 'b')
				{
					blue_score += values[y + 1][x];
					green_score -= values[y + 1][x];
				}
				else
				{
					green_score += values[y + 1][x];
					blue_score -= values[y + 1][x];
				}
			}
			if (x > 0 && game[y][x - 1] == opponent)
			{
				game[y][x - 1] = current_team;
				if (current_team == 'b')
				{
					blue_score += values[y][x - 1];
					green_score -= values[y][x - 1];
				}
				else
				{
					green_score += values[y][x - 1];
					blue_score -= values[y][x - 1];
				}
			}
			if (x < GAME_DIMENSION - 1 && game[y][x + 1] == opponent)
			{
				game[y][x + 1] = current_team;
				if (current_team == 'b')
				{
					blue_score += values[y][x + 1];
					green_score -= values[y][x + 1];
				}
				else
				{
					green_score += values[y][x + 1];
					blue_score -= values[y][x + 1];
				}
			}
		}
		if (current_team == 'b')
		{
			current_team = 'g';
			opponent = 'b';
		}
		else
		{
			current_team = 'b';
			opponent = 'g';
		}
	}
	//memory cleanup
	for (int i = 0; i < GAME_DIMENSION; i++)
	{
		delete[] game_copy[i];
	}
	delete[] game_copy;

}

int max_val(char** game_board, char Max_team, char Min_team, int depth, int& x, int& y)
{
	int best = -10000; //best value so far is held here
	int bestKernel = -10000;
	for (int i = 0; i < GAME_DIMENSION; i++)
	{
		for (int j = 0; j < GAME_DIMENSION; j++)
		{
			//check if we have reached a terminal node
			if (blocks_occupied + depth - 1 == GAME_DIMENSION*GAME_DIMENSION - 1)
			{
				if (game_board[i][j] == 'o')
				{
					//set location values which will be sent back
					x = j;
					y = i;
					int utility = values[i][j];
					//check for neighbors
					if ((i > 0 && game_board[i - 1][j] == Max_team) || (i < GAME_DIMENSION - 1 && game_board[i + 1][j] == Max_team) || (j > 0 && game_board[i][j - 1] == Max_team) || (j < GAME_DIMENSION - 1 && game_board[i][j + 1] == Max_team))
					{
						if (i > 0 && game_board[i - 1][j] == Min_team)
						{
							utility += values[i - 1][j]*2;
						}
						if (i < GAME_DIMENSION - 1 && game_board[i + 1][j] == Min_team)
						{
							utility += values[i + 1][j]*2;
						}
						if (j > 0 && game_board[i][j - 1] == Min_team)
						{
							utility += values[i][j - 1]*2;
						}
						if (j < GAME_DIMENSION - 1 && game_board[i][j + 1] == Min_team)
						{
							utility += values[i][j + 1]*2;
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
				if (game_board[i][j] == 'o')
				{
					//MAKE COPY EACH TIME
					char** copy = new char*[GAME_DIMENSION];
					for (int k = 0; k < GAME_DIMENSION; k++)
					{
						copy[k] = new char[GAME_DIMENSION];
					}
					for (int k = 0; k < GAME_DIMENSION; k++)
					{
						for (int m = 0; m < GAME_DIMENSION; m++)
						{
							copy[k][m] = game_board[k][m];
						}
					}
					//perform para drop
					copy[i][j] = Max_team;
					//check for neighbors
					if ((i > 0 && copy[i - 1][j] == Max_team) || (i < GAME_DIMENSION - 1 && copy[i + 1][j] == Max_team) || (j > 0 && copy[i][j - 1] == Max_team) || (j < GAME_DIMENSION - 1 && copy[i][j + 1] == Max_team))
					{
						if (i > 0 && copy[i - 1][j] == Min_team)
						{
							copy[i - 1][j] = Max_team;
						}
						if (i < GAME_DIMENSION - 1 && copy[i + 1][j] == Min_team)
						{
							copy[i + 1][j] = Max_team;
						}
						if (j > 0 && copy[i][j - 1] == Min_team)
						{
							copy[i][j - 1] = Max_team;
						}
						if (j < GAME_DIMENSION - 1 && copy[i][j + 1] == Min_team)
						{
							copy[i][j + 1] = Max_team;
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
				if (game_board[i][j] == 'o')
				{
					if (pass == 1)
					{
						//MAKE COPY EACH TIME
						//int local_best;
						char** copy = new char*[GAME_DIMENSION];
						for (int k = 0; k < GAME_DIMENSION; k++)
						{
							copy[k] = new char[GAME_DIMENSION];
						}
						for (int k = 0; k < GAME_DIMENSION; k++)
						{
							for (int m = 0; m < GAME_DIMENSION; m++)
							{
								copy[k][m] = game_board[k][m];
							}
						}
						//perform para drop
						copy[i][j] = Max_team;
						//check for neighbors
						if ((i > 0 && copy[i - 1][j] == Max_team) || (i < GAME_DIMENSION - 1 && copy[i + 1][j] == Max_team) || (j > 0 && copy[i][j - 1] == Max_team) || (j < GAME_DIMENSION - 1 && copy[i][j + 1] == Max_team))
						{
							if (i > 0 && copy[i - 1][j] == Min_team)
							{
								copy[i - 1][j] = Max_team;
							}
							if (i < GAME_DIMENSION - 1 && copy[i + 1][j] == Min_team)
							{
								copy[i + 1][j] = Max_team;
							}
							if (j > 0 && copy[i][j - 1] == Min_team)
							{
								copy[i][j - 1] = Max_team;
							}
							if (j < GAME_DIMENSION - 1 && copy[i][j + 1] == Min_team)
							{
								copy[i][j + 1] = Max_team;
							}
						}
						/*copy into host array so device can handle the next level*/

						for (int ii = 0; ii < GAME_DIMENSION; ii++)
						{
							for (int jj = 0; jj < GAME_DIMENSION; jj++)
							{
								//hostArray[arrayCount*GAME_DIMENSION*GAME_DIMENSION + GAME_DIMENSION*ii + jj] = copy[ii][jj]; // no memory coalescing
								hostArray[currentSize*(ii*GAME_DIMENSION + jj) + arrayCount] = copy[ii][jj];
							}
						}
						arrayCount++;
						//memory cleanup
						for (int k = 0; k < GAME_DIMENSION; k++)
						{
							delete[] copy[k];
						}
						delete[] copy;
					}
					/*second pass so read in values from memory filled by kernel, set x and y coordinates, and return value*/
					else
					{
						int localKernel = hostOutput[arrayCount * 3];
						//cout << bestKernel << " " << x << " " << y << "    ";
						if (localKernel > bestKernel)
						{
							bestKernel = localKernel;
							y = hostOutput[arrayCount * 3 + 1];
							x = hostOutput[arrayCount * 3 + 2];
						}
						arrayCount++;
					}
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
		return bestKernel;
	}

}

int min_val(char** game_board, char Max_team, char Min_team, int depth, int& x, int& y)
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
				if (game_board[i][j] == 'o')
				{
					//set location values which will be sent back
					x = j;
					y = i;
					int utility = values[i][j];
					//check for neighbors
					if ((i > 0 && game_board[i - 1][j] == Min_team) || (i < GAME_DIMENSION - 1 && game_board[i + 1][j] == Min_team) || (j > 0 && game_board[i][j - 1] == Min_team) || (j < GAME_DIMENSION - 1 && game_board[i][j + 1] == Min_team))
					{
						if (i > 0 && game_board[i - 1][j] == Max_team)
						{
							utility += values[i - 1][j]*2;
						}
						if (i < GAME_DIMENSION - 1 && game_board[i + 1][j] == Max_team)
						{
							utility += values[i + 1][j]*2;
						}
						if (j > 0 && game_board[i][j - 1] == Max_team)
						{
							utility += values[i][j - 1]*2;
						}
						if (j < GAME_DIMENSION - 1 && game_board[i][j + 1] == Max_team)
						{
							utility += values[i][j + 1]*2;
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
				if (game_board[i][j] == 'o')
				{
					//MAKE COPY EACH TIME
					char** copy = new char*[GAME_DIMENSION];
					for (int k = 0; k < GAME_DIMENSION; k++)
					{
						copy[k] = new char[GAME_DIMENSION];
					}
					for (int k = 0; k < GAME_DIMENSION; k++)
					{
						for (int m = 0; m < GAME_DIMENSION; m++)
						{
							copy[k][m] = game_board[k][m];
						}
					}
					//perform para drop
					copy[i][j] = Min_team;
					//check for neighbors
					if ((i > 0 && copy[i - 1][j] == Min_team) || (i < GAME_DIMENSION - 1 && copy[i + 1][j] == Min_team) || (j > 0 && copy[i][j - 1] == Min_team) || (j < GAME_DIMENSION - 1 && copy[i][j + 1] == Min_team))
					{
						if (i > 0 && copy[i - 1][j] == Max_team)
						{
							copy[i - 1][j] = Min_team;
						}
						if (i < GAME_DIMENSION - 1 && copy[i + 1][j] == Max_team)
						{
							copy[i + 1][j] = Min_team;
						}
						if (j > 0 && copy[i][j - 1] == Max_team)
						{
							copy[i][j - 1] = Min_team;
						}
						if (j < GAME_DIMENSION - 1 && copy[i][j + 1] == Max_team)
						{
							copy[i][j + 1] = Min_team;
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
				if (game_board[i][j] == 'o')
				{
					//MAKE COPY EACH TIME
					int local_best;
					char** copy = new char*[GAME_DIMENSION];
					for (int k = 0; k < GAME_DIMENSION; k++)
					{
						copy[k] = new char[GAME_DIMENSION];
					}
					for (int k = 0; k < GAME_DIMENSION; k++)
					{
						for (int m = 0; m < GAME_DIMENSION; m++)
						{
							copy[k][m] = game_board[k][m];
						}
					}
					//perform para drop
					copy[i][j] = Min_team;
					//check for neighbors
					if ((i > 0 && copy[i - 1][j] == Min_team) || (i < GAME_DIMENSION - 1 && copy[i + 1][j] == Min_team) || (j > 0 && copy[i][j - 1] == Min_team) || (j < GAME_DIMENSION - 1 && copy[i][j + 1] == Min_team))
					{
						if (i > 0 && copy[i - 1][j] == Max_team)
						{
							copy[i - 1][j] = Min_team;
						}
						if (i < GAME_DIMENSION - 1 && copy[i + 1][j] == Max_team)
						{
							copy[i + 1][j] = Min_team;
						}
						if (j > 0 && copy[i][j - 1] == Max_team)
						{
							copy[i][j - 1] = Min_team;
						}
						if (j < GAME_DIMENSION - 1 && copy[i][j + 1] == Max_team)
						{
							copy[i][j + 1] = Min_team;
						}
					}
					/*add up all current values on board of max_team and min_team and compute the difference*/
					for (int k = 0; k < GAME_DIMENSION; k++)
					{
						for (int m = 0; m < GAME_DIMENSION; m++)
						{
							if (copy[k][m] == Max_team)
							{
								max_total += values[k][m];
							}
							else if (copy[k][m] == Min_team)
							{
								min_total += values[k][m];
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

/*initializes each block in the game*/
void setup_game(int x)
{
	int deviceValues[GAME_DIMENSION*GAME_DIMENSION];
	for (int i = 0; i < GAME_DIMENSION; i++)
	{
		for (int j = 0; j < GAME_DIMENSION; j++)
		{
			values[i][j] = gameboard[x][i][j]; //set values for cpu
			game[i][j] = 'o';
			deviceValues[i*GAME_DIMENSION + j] = gameboard[x][i][j];
		}
	}
	//set constant memory
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpyToSymbol(Vc, deviceValues, GAME_DIMENSION*GAME_DIMENSION*sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpyToSymbol failed!");
	}
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
		outFile << "Player Blue took " << blue_time << " milliseconds (" << float(blue_time) / float(GAME_DIMENSION*GAME_DIMENSION / 2) << "ms per move)" << endl;
		outFile << "Player Green took " << green_time << " milliseconds (" << float(green_time) / float(GAME_DIMENSION*GAME_DIMENSION / 2) << "ms per move)" << endl;
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
				if (game[i][j] == 'b')
				{
					outFile << 'B';
					if (game_mode != AI)
						cout << 'B';
				}
				else if (game[i][j] == 'g')
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

int main()
{
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	size_t size = 1024 * 1024 * 1024; //set a good amount of memory
	cudaStatus = cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetLimit failed!  Do you have a CUDA-capable GPU installed?");
	}
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

	game = new char*[GAME_DIMENSION]; //pointer to gameboard
	for (int i = 0; i < GAME_DIMENSION; i++)
	{
		game[i] = new char[GAME_DIMENSION];
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
	cout << kernelLaunchCount << endl;
	cout << "Process Execution Time " << programTime << "s" << endl; //output time taken to run program
	cin.ignore();
	cin.get(); // pause program to view results
	return 0;
}