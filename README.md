# nimCUDA
Simulation of the game of Nim using CUDA

# Game Description
The game of Nim is a mathematical strategy game where 2 players take turns removing sticks
from a single pile, with the goal of forcing the opponent to take the last item.
The player who takes the last stick is the winner. The other player is the loser.

# Technical Notes and Considerations
Each player randomly picks a number of sticks from the pile. 

The code is meant to illustrate a basic knowledge of CUDA coding by creating 2 player kernels
that randomly pick up sticks on a multi GPU system by interrogating the API to determine
how many GPUs are present. If only 1 GPU is present the implementation fallback uses CUDA streams
per player. The number of sticks in the pile is betweeen 10 and 100.

# How to build and run from Makefile
```cd nimCUDA
make
./nim.exe
```

# or to manually build and run
```nvcc --std c++17 nim.cu -o nim.exe
./nim.exe
```
