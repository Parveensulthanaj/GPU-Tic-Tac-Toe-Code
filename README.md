# GPU vs GPU Tic-Tac-Toe Using CUDA
# Author

Name: PARVEEN SULTHANA J
Course: Parallel Computing / CUDA Programming  
Project: GPU vs GPU Tic-Tac-Toe Using CUDA

## Project Overview

GPU vs GPU Tic-Tac-Toe Using CUDA is a parallel computing project where two GPU-based AI players compete in a Tic-Tac-Toe game using CUDA kernels for parallel move execution and decision-making.

The project demonstrates how CUDA programming and GPU parallelism can be used to implement game logic and AI strategies.

---

# Features

- GPU vs GPU gameplay
- CUDA kernel-based move selection
- Parallel execution using GPU threads
- Smart AI vs Random AI
- Real-time board visualization
- Winner and draw detection
- Simple and easy-to-understand implementation

---

# Technologies Used

- CUDA C++
- NVIDIA CUDA Toolkit
- Google Colab / Linux / Windows
- GPU Parallel Computing

---

# Game Description

The game is played on a 3×3 Tic-Tac-Toe board between two GPU competitors.

## GPU 1 (Player X)
- Uses a random strategy
- Selects available positions using GPU threads

## GPU 2 (Player O)
- Uses a smarter strategy
- Attempts to:
  1. Win the game
  2. Block the opponent
  3. Choose fallback empty positions

The game continues until:
- Player X wins
- Player O wins
- Match ends in a draw

---

# CUDA Concepts Used

This project demonstrates several CUDA programming concepts:

- CUDA Kernels
- Thread Parallelism
- Device Memory Allocation
- Host-to-Device Memory Transfer
- Device-to-Host Memory Transfer
- Atomic Operations
- GPU Synchronization

---

# File Structure

```text
gpu_tictactoe.cu
README.md
```

---

# How to Run in Google Colab

## Step 1: Open Google Colab

https://colab.research.google.com/

---

## Step 2: Enable GPU

- Click Runtime
- Click Change Runtime Type
- Select Hardware Accelerator → GPU
- Click Save

---

## Step 3: Check GPU

Run:

```python
!nvidia-smi
```

---

## Step 4: Create CUDA File

```python
%%writefile gpu_tictactoe.cu



#include <iostream>
#include <cuda.h>
#include <cstdlib>
#include <ctime>

using namespace std;

#define EMPTY 0
#define PLAYER_X 1
#define PLAYER_O 2

// Print the game board
void printBoard(int board[9])
{
    cout << "\n";

    for(int i = 0; i < 9; i++)
    {
        char c;

        if(board[i] == PLAYER_X)
            c = 'X';
        else if(board[i] == PLAYER_O)
            c = 'O';
        else
            c = '-';

        cout << c << " ";

        if((i + 1) % 3 == 0)
            cout << endl;
    }

    cout << endl;
}

// Check winner
int checkWinner(int board[9])
{
    int wins[8][3] = {
        {0,1,2},
        {3,4,5},
        {6,7,8},
        {0,3,6},
        {1,4,7},
        {2,5,8},
        {0,4,8},
        {2,4,6}
    };

    for(int i = 0; i < 8; i++)
    {
        int a = wins[i][0];
        int b = wins[i][1];
        int c = wins[i][2];

        if(board[a] != EMPTY &&
           board[a] == board[b] &&
           board[b] == board[c])
        {
            return board[a];
        }
    }

    return 0;
}

// Check draw
bool isDraw(int board[9])
{
    for(int i = 0; i < 9; i++)
    {
        if(board[i] == EMPTY)
            return false;
    }

    return true;
}

// GPU Kernel for Random Player (GPU 1)
__global__ void randomMoveKernel(int *board, int *move)
{
    int idx = threadIdx.x;

    if(idx < 9)
    {
        if(board[idx] == EMPTY)
        {
            atomicMin(move, idx);
        }
    }
}

// GPU Kernel for Smart Player (GPU 2)
__global__ void smartMoveKernel(int *board, int *move)
{
    int wins[8][3] = {
        {0,1,2},
        {3,4,5},
        {6,7,8},
        {0,3,6},
        {1,4,7},
        {2,5,8},
        {0,4,8},
        {2,4,6}
    };

    int tid = threadIdx.x;

    if(tid < 8)
    {
        int a = wins[tid][0];
        int b = wins[tid][1];
        int c = wins[tid][2];

        // Try winning move
        if(board[a] == PLAYER_O &&
           board[b] == PLAYER_O &&
           board[c] == EMPTY)
        {
            *move = c;
        }

        else if(board[a] == PLAYER_O &&
                board[c] == PLAYER_O &&
                board[b] == EMPTY)
        {
            *move = b;
        }

        else if(board[b] == PLAYER_O &&
                board[c] == PLAYER_O &&
                board[a] == EMPTY)
        {
            *move = a;
        }

        // Block opponent
        else if(board[a] == PLAYER_X &&
                board[b] == PLAYER_X &&
                board[c] == EMPTY)
        {
            *move = c;
        }

        else if(board[a] == PLAYER_X &&
                board[c] == PLAYER_X &&
                board[b] == EMPTY)
        {
            *move = b;
        }

        else if(board[b] == PLAYER_X &&
                board[c] == PLAYER_X &&
                board[a] == EMPTY)
        {
            *move = a;
        }
    }
}

int main()
{
    srand(time(0));

    int board[9] = {0};

    int *d_board;
    int *d_move;

    cudaMalloc((void**)&d_board, 9 * sizeof(int));
    cudaMalloc((void**)&d_move, sizeof(int));

    int turn = PLAYER_X;

    cout << "==============================" << endl;
    cout << " GPU vs GPU Tic-Tac-Toe Game " << endl;
    cout << "==============================" << endl;

    while(true)
    {
        cudaMemcpy(d_board,
                   board,
                   9 * sizeof(int),
                   cudaMemcpyHostToDevice);

        int move = 100;

        cudaMemcpy(d_move,
                   &move,
                   sizeof(int),
                   cudaMemcpyHostToDevice);

        // GPU 1 -> Random
        if(turn == PLAYER_X)
        {
            randomMoveKernel<<<1,9>>>(d_board, d_move);

            cudaDeviceSynchronize();

            cudaMemcpy(&move,
                       d_move,
                       sizeof(int),
                       cudaMemcpyDeviceToHost);

            cout << "GPU 1 (X) chooses position: "
                 << move << endl;
        }

        // GPU 2 -> Smart
        else
        {
            smartMoveKernel<<<1,8>>>(d_board, d_move);

            cudaDeviceSynchronize();

            cudaMemcpy(&move,
                       d_move,
                       sizeof(int),
                       cudaMemcpyDeviceToHost);

            // fallback move
            if(move == 100)
            {
                for(int i = 0; i < 9; i++)
                {
                    if(board[i] == EMPTY)
                    {
                        move = i;
                        break;
                    }
                }
            }

            cout << "GPU 2 (O) chooses position: "
                 << move << endl;
        }

        // Apply move
        if(move >= 0 &&
           move < 9 &&
           board[move] == EMPTY)
        {
            board[move] = turn;
        }

        // Display board
        printBoard(board);

        // Check winner
        int winner = checkWinner(board);

        if(winner == PLAYER_X)
        {
            cout << "GPU 1 (X) WINS!" << endl;
            break;
        }

        if(winner == PLAYER_O)
        {
            cout << "GPU 2 (O) WINS!" << endl;
            break;
        }

        // Check draw
        if(isDraw(board))
        {
            cout << "MATCH DRAW!" << endl;
            break;
        }

        // Switch turns
        if(turn == PLAYER_X)
            turn = PLAYER_O;
        else
            turn = PLAYER_X;
    }

    cudaFree(d_board);
    cudaFree(d_move);

    return 0;
}
```

---

## Step 5: Compile CUDA Program

```python
!nvcc gpu_tictactoe.cu -o game
```

---

## Step 6: Run Program

```python
!./game
```

---

# Compilation for Local System

## Linux

```bash
nvcc gpu_tictactoe.cu -o game
./game
```

## Windows

```bash
nvcc gpu_tictactoe.cu -o game.exe
game.exe
```

---
# Kernel Description

## randomMoveKernel()

- Executed by GPU 1
- Finds available empty cells
- Uses atomic operations to select moves

## smartMoveKernel()

- Executed by GPU 2
- Checks winning combinations
- Attempts to:
  - Win
  - Block opponent
  - Select valid move

---

# Functions Used

| Function | Purpose |
|----------|----------|
| printBoard() | Displays board |
| checkWinner() | Detects winner |
| isDraw() | Detects draw |
| randomMoveKernel() | Random GPU strategy |
| smartMoveKernel() | Smart GPU strategy |

---

# Advantages of the Project

- Demonstrates CUDA programming basics
- Shows GPU parallel execution
- Easy visualization of GPU-based AI
- Beginner-friendly CUDA project
- Good example of parallel game processing

---

# Future Improvements

- Multi-GPU support
- Connect 4 implementation
- Chess AI using CUDA
- Minimax algorithm on GPU
- CUDA Streams for asynchronous gameplay
- GUI-based visualization

---

# Presentation Points

## Project Description
- Explain GPU vs GPU architecture
- Explain AI strategies

## Code Description
- Explain CUDA kernels
- Explain memory transfer
- Explain parallel threads

## Demonstration
- Show gameplay
- Show board updates
- Explain winner detection

---


# Example Output
<img width="186" height="353" alt="GPU1" src="https://github.com/user-attachments/assets/9a010234-df8a-4d3c-8e33-f64d189ded40" />
<img width="159" height="168" alt="GPU2" src="https://github.com/user-attachments/assets/6a3041d2-be0e-4ee9-808b-408a8fd52381" />


# Conclusion

This project successfully demonstrates how CUDA and GPU parallel computing can be used to build a simple AI-based game. The implementation highlights GPU kernel execution, memory management, and parallel decision-making in an interactive Tic-Tac-Toe game.

---
