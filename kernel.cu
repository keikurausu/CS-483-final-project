
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
#define THREADS 256

//@@ Define constant memory for device kernel here
__constant__ int Vc[GAME_DIMENSION][GAME_DIMENSION]; //holds gameboard values for device

enum gameMode {AI, HUMAN, DOUBLE_HUMAN};

int values[GAME_DIMENSION][GAME_DIMENSION]; //constant array of gameboard values which is filled based on selected map

// 'o' for open, 'b' for blue, 'g' for green team

/*function declarations*/
void setup_game(int x);
void output_game(string filename);
void play_game();
int max_val(char** game_board, char Max_team, char Min_team, int depth, int& x, int& y);
int min_val(char** game_board, char Max_team, char Min_team, int depth, int& x, int& y);
cudaError_t searchHelper(char* input, const int size);

char** game;  //pointer to gameboard
char* hostArray; // array which holds stuff that needs to be copied to device
int* hostOutput = 0; //pointer to host output array
int arrayCount;

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

//input holds game states up to depth 3, output will hold 3 values to be evaluated at each depth 3 node as follows. [utility value][y-coord][x-coord]
__global__ void gameSearchKernel(char* input, int* output, const int size)
{
	//int i = threadIdx.x;
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		printf("%d %d %d %d %d %d %d %d %d %d", Vc[0][1], Vc[0][2], Vc[0][3], Vc[0][4], Vc[0][5], Vc[1][0], Vc[1][1], Vc[1][2], Vc[1][3], Vc[1][4]);
	}
	output[blockIdx.x*blockDim.x + threadIdx.x] = 1;
}

// Helper function for launching GPU --size is number of game boards
cudaError_t searchHelper(char* input, const int size)
{
	char* dev_input = 0;
	int* dev_output = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	//allocate host memory
	cudaMallocHost(&hostOutput, 3 * size * sizeof(int));
	// Allocate GPU memory  .
	cudaStatus = cudaMalloc((void**)&dev_output, 3 * size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc device output failed!");
		goto Error;
	}
	cudaStatus = cudaMemset(dev_output, 0, 3*size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemset device output failed!");
		goto Error;
	}
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

	//set up launch parameters
	dim3 grid((size - 1) / THREADS + 1, 1, 1);
	dim3 threads(THREADS, 1, 1);

	// Launch kernel
	gameSearchKernel << <grid, threads >> >(dev_input, dev_output, size);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("gameSearchKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching gameSearchKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(hostOutput, dev_output, 3* size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy 2 device to host failed!");
		goto Error;
	}

Error:
	cudaFree(dev_input);
	cudaFree(dev_output);
	cout << hostOutput[1];
	return cudaStatus;
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
	cudaMemcpyToSymbol(Vc, deviceValues, GAME_DIMENSION*GAME_DIMENSION*sizeof(int));
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
		outFile << "Player Blue took " << blue_time << " milliseconds (" << float(blue_time) / float(GAME_DIMENSION*GAME_DIMENSION/2) << "ms per move)" << endl;
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
					if (game_mode!=AI)
						cout << 'B';
				}
				else if (game[i][j] == 'b')
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
			if (GAME_DIMENSION*GAME_DIMENSION - blocks_occupied >= CPU_END_LIMIT)
			{
				hostArray = new char[(GAME_DIMENSION*GAME_DIMENSION - blocks_occupied)*(GAME_DIMENSION*GAME_DIMENSION - blocks_occupied - 1)*(GAME_DIMENSION*GAME_DIMENSION - blocks_occupied - 2)*GAME_DIMENSION*GAME_DIMENSION]; //allocate memory to store data that needs to be copied to device
				arrayCount = 0;
				double start = clock();
				cudaError_t cudaStatus = searchHelper(hostArray, (GAME_DIMENSION*GAME_DIMENSION - blocks_occupied)*(GAME_DIMENSION*GAME_DIMENSION - blocks_occupied - 1)*(GAME_DIMENSION*GAME_DIMENSION - blocks_occupied - 2));
				if (cudaStatus != cudaSuccess) {
					printf("searchHelper failed!");
				}
				// cudaDeviceReset must be called before exiting in order for profiling and
				// tracing tools such as Nsight and Visual Profiler to show complete traces.
				cudaStatus = cudaDeviceReset();
				if (cudaStatus != cudaSuccess) {
					printf("cudaDeviceReset failed!");
				}
				//max_val(game_copy, current_team, opponent, 1, x, y); //take turn -- once this function returns x and y will hold location of where to go next
				double turn_time = (clock() - start);
				blue_time += turn_time;
				delete[] hostArray; // free array

			}
			else{
				double start = clock();
				max_val(game_copy, current_team, opponent, 1, x, y); //take turn -- once this function returns x and y will hold location of where to go next
				double turn_time = (clock() - start);
				blue_time += turn_time;
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
			if (GAME_DIMENSION*GAME_DIMENSION - blocks_occupied >= CPU_END_LIMIT)
			{
				hostArray = new char[(GAME_DIMENSION*GAME_DIMENSION - blocks_occupied)*(GAME_DIMENSION*GAME_DIMENSION - blocks_occupied - 1)*(GAME_DIMENSION*GAME_DIMENSION - blocks_occupied - 2)*GAME_DIMENSION*GAME_DIMENSION]; //allocate memory to store data that needs to be copied to device
				arrayCount = 0;
				double start = clock();
				max_val(game_copy, current_team, opponent, 1, x, y); //take turn -- once this function returns x and y will hold location of where to go next
				double turn_time = (clock() - start);
				blue_time += turn_time;
				delete[] hostArray; // free array

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
			} while (game[human_y-1][human_x-1] != 'o');
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
		cudaFreeHost(hostOutput); //free host memory
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
	int best = -1000; //best value so far is held here
	int best_evaluation = -1000; //used for evaluation function

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
					if (GAME_DIMENSION*GAME_DIMENSION - blocks_occupied >= CPU_END_LIMIT)
					{
						for (int ii = 0; ii < GAME_DIMENSION; ii++)
						{
							for (int jj = 0; jj < GAME_DIMENSION; jj++)
							{
								hostArray[arrayCount*GAME_DIMENSION*GAME_DIMENSION + GAME_DIMENSION*ii + jj] = copy[ii][jj];
							}
						} 
						arrayCount++;
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
	cout << "Process Execution Time " << programTime << "s" << endl;
	cin.ignore();
	cin.get(); // pause program to view results
	return 0;
}