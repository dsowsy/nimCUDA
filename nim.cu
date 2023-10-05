#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <ctime>
#include <cstdlib>

// 
//  Programmer: David Sowsy
//      
//  Game Description: 
//    The game of Nim is a mathematical strategy game where 2 players take turns removing sticks
//    from a single pile, with the goal of forcing the opponent to take the last item.
//    The player who takes the last stick is the winner. The other player is the loser.
// 
//  Notes: 
//    This code maps the kernel to 2 GPUs if more than 1 GPU is present.
//    If only 1 GPU is present, each player is run on a separate stream.
//
//  Revision History: 
//      Initial ideation                          10-02-2023 
//      Initial coding                            10-04-2023
//      Consideration of GPU to stream fallback   10-05-2023
//      Additional output & turn enforcement      10-05-2023

// Helper functions for CUDA randomization
__device__ int GenerateRandomNumber(int max_value, curandState_t* state) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int move = curand(&state[id]) % max_value + 1;  // Random move
  return move;
}

__global__ void InitializeCurand(curandState_t* state) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(1234 + id, id, 0, &state[id]);
}

// First player
__global__ void Player1(int* pile, curandState_t* state, int* sticks_taken) {
  int move = GenerateRandomNumber(*pile, state);
  if (*pile > 0) {
    *pile -= move;
    *sticks_taken = move;
  }
}

// Second player
__global__ void Player2(int* pile, curandState_t* state, int* sticks_taken) {
  int move = GenerateRandomNumber(*pile, state);
  if (*pile > 0) {
    *pile -= move;
    *sticks_taken = move; 
  }
}

int main() {
  srand(time(0));  // Seed for random number generation

  int num_gpus = 0;
  cudaGetDeviceCount(&num_gpus);  // Get the number of available GPUs

  // Setup game state with a random pile size greater than 10
  int pile_size = rand() % 90 + 10;  // Random number between 10 and 100
  int *d_pile1, *d_pile2;  // Device pointers

  // Track the number of sticks taken back from the kernel.
  int *d_sticks_taken1, *d_sticks_taken2;
  cudaMalloc(&d_sticks_taken1, sizeof(int));
  cudaMalloc(&d_sticks_taken2, sizeof(int));

  // Allocate memory on the GPU
  cudaSetDevice(0);  // Set the first GPU as the current device
  cudaMalloc(&d_pile1, sizeof(int));
  cudaMemcpy(d_pile1, &pile_size, sizeof(int), cudaMemcpyHostToDevice);

  curandState_t* d_state1;
  cudaMalloc(&d_state1, sizeof(curandState_t));
  InitializeCurand<<<1, 1>>>(d_state1);

  curandState_t* d_state2;
  cudaMalloc(&d_state2, sizeof(curandState_t));
  InitializeCurand<<<1, 1>>>(d_state2);
  
  if (num_gpus > 1) {
    cudaSetDevice(1);  // Set the second GPU as the current device
    cudaMalloc(&d_pile2, sizeof(int));
    cudaMemcpy(d_pile2, &pile_size, sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "There are " << num_gpus << " present. Executing GPU strategy." << std::endl;
  } else {
    std::cout << "There is only 1 GPU present. Executing streams strategy." << std::endl;
  }

  std::cout << "Initial pile size is " << pile_size << "."<< std::endl;

  int player_turn = 1; 
  // Main game loop
  while (pile_size > 0) {
    // Call player kernels alternatively
    int sticks_taken; 
    if (player_turn == 1) {
      if (num_gpus > 1) {
        Player1<<<1, 1>>>(d_pile1, d_state1, d_sticks_taken1);
        cudaDeviceSynchronize();
        cudaSetDevice(1);  // Switch to the second GPU for Player 2
      } else {
        cudaStream_t stream_player1;
        cudaStreamCreate(&stream_player1);
        Player1<<<1, 1, 0, stream_player1>>>(d_pile1, d_state1, d_sticks_taken1);
        cudaDeviceSynchronize();
        cudaStreamDestroy(stream_player1);
      }
      cudaMemcpy(&sticks_taken, d_sticks_taken1, sizeof(int), cudaMemcpyDeviceToHost);
      std::cout << "Player 1 takes " << sticks_taken << " stick(s). ";
      player_turn = 2;
    } else {
      if (num_gpus > 1) {
        Player2<<<1, 1>>>(d_pile2, d_state2, d_sticks_taken2);
        cudaDeviceSynchronize();
        cudaSetDevice(0);  // Switch back to the first GPU for Player 1
      } else {
        cudaStream_t stream_player2;
        cudaStreamCreate(&stream_player2);
        Player2<<<1, 1, 0, stream_player2>>>(d_pile1, d_state1, d_sticks_taken2);
        cudaDeviceSynchronize();
        cudaStreamDestroy(stream_player2);
      }
      cudaMemcpy(&sticks_taken, d_sticks_taken2, sizeof(int), cudaMemcpyDeviceToHost);
      std::cout << "Player 2 takes " << sticks_taken << " stick(s). ";
      player_turn = 1;
    }

    // Copy updated pile_size from GPU to CPU
    cudaMemcpy(&pile_size, (num_gpus > 1) ? (pile_size % 2 == 0 ? d_pile1 : d_pile2) : d_pile1, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << pile_size << " stick(s) remain.\n";

    // Check for win condition
    if (pile_size == 0) {
      std::cout << (pile_size % 2 == 0 ? "Player 2" : "Player 1") << " wins!\n";
      break;
    }
  }

  // Free GPU memory
  cudaSetDevice(0);
  cudaFree(d_pile1);
  cudaFree(d_state1);
  if (num_gpus > 1) {
    cudaSetDevice(1);
    cudaFree(d_pile2);
    cudaFree(d_state2);
  }

  cudaFree(d_sticks_taken1);
  cudaFree(d_sticks_taken2);
  return 0;
}

