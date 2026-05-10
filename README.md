# Advanced Parallel Grid Processing with MPI

A parallel distributed processing system built with **MPI in C++** that processes 2D grid and matrix data using parallel decomposition and inter-process communication.

> **Course:** Parallel Computing — Faculty of Computers and Artificial Intelligence, Fayoum University (2026)

---

## Algorithms

### 🧬 Game of Life (Category A — Grid/Spatial)
Conway's Game of Life simulated in parallel across a distributed grid.
- Load any starting pattern from a `.txt` file
- Watch the grid evolve iteration by iteration directly in the terminal
- Supports classic patterns: Glider, Blinker, Block, Toad, Beacon

### ✖️ Matrix Multiplication (Category B — Data/Computation)
Parallel matrix multiplication using a Master-Worker pattern with dynamic load balancing.
- Load large matrices from files
- Works with any matrix size and any number of processes
- Coordinator dispatches row chunks dynamically — faster workers get more work

---

## Communication Strategies

| Strategy | Where Used |
|---|---|
| Blocking `MPI_Send` / `MPI_Recv` | Matrix Multiplication — chunk dispatch and result collection |
| Non-blocking `MPI_Isend` / `MPI_Irecv` | Game of Life — ghost row exchange with computation overlap |
| Collective `MPI_Bcast` / `MPI_Scatterv` / `MPI_Gatherv` | Both algorithms — data distribution and gathering |
| `MPI_Comm_split` | Both algorithms — process organization into logical groups |

---

## Project Structure

```
├── main.cpp          # Entry point, menu loop, input validation
├── config.h          # Shared structs: Matrix, IntMatrix, AlgorithmType
├── io_utils.h        # I/O function declarations
├── io_utils.cpp      # File reading/writing with error handling
├── mpi_utils.h       # Algorithm function declarations
├── mpi_utils.cpp     # Parallel algorithm implementations
└── patterns/
    ├── glider.txt
    ├── blinker.txt
    ├── block.txt
    ├── toad.txt
    └── beacon.txt
```

---

## Requirements

- C++ compiler with MPI support
- [Microsoft MPI](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi) (Windows) or [OpenMPI](https://www.open-mpi.org/) (Linux/Mac)
- At least **2 MPI processes** to run

---

## Build

### Windows (Visual Studio + MS-MPI)
Open the solution in Visual Studio, set configuration to `Release` or `Debug`, and build.

### Linux (GCC + OpenMPI)
```bash
mpic++ -O2 -o parallel_project main.cpp io_utils.cpp mpi_utils.cpp
```

---

## Run

```bash
mpiexec -n <num_processes> ./parallel_project
```

Examples:
```bash
mpiexec -n 4  ./parallel_project   # 4 processes
mpiexec -n 8  ./parallel_project   # 8 processes
mpiexec -n 16 ./parallel_project   # 16 processes
```

> Minimum: 2 processes. Recommended: 4 or more for meaningful parallelism.

---

## Usage

When you run the program you get an interactive menu:

```
============================
MPI Parallel Processing System
Running on 4 processes
============================
Choose Algorithm:
1. Game of Life
2. Matrix Multiplication
3. Quit
>
```

The program keeps running until you choose **Quit**, so you can run multiple algorithms in one session.

### Game of Life
```
> 1
Enter pattern file name (e.g. glider.txt):
> glider.txt
Enter number of iterations:
> 10
```
The grid is printed to the terminal after every iteration.

### Matrix Multiplication
```
> 2
Enter file name for Matrix A:
> matA.txt
Enter file name for Matrix B:
> matB.txt
```

---

## Pattern File Format

Game of Life pattern files are plain text grids of `0` (dead) and `1` (alive):

```
0 0 0 0 0
0 0 1 0 0
0 0 0 1 0
0 1 1 1 0
0 0 0 0 0
```

### Included Patterns

| File | Pattern | Behavior |
|---|---|---|
| `glider.txt` | Glider | Moves diagonally across the grid |
| `blinker.txt` | Blinker | Oscillates between horizontal and vertical |
| `block.txt` | Block | Stable — never changes |
| `toad.txt` | Toad | Oscillates with period 2 |
| `beacon.txt` | Beacon | Oscillates with period 2 |

---

## Matrix File Format

Plain text, space-separated values, one row per line:

```
1 2 3
4 5 6
7 8 9
```

### Generate a Large Test Matrix (Python)

```python
import random
rows, cols = 1000, 1000
with open("big.txt", "w") as f:
    for i in range(rows):
        f.write(" ".join(str(random.randint(1, 10))
                for _ in range(cols)) + "\n")
```

---

## Error Handling

The system handles all invalid inputs gracefully without crashing:

| Error | Behavior |
|---|---|
| Wrong menu choice (not 1/2/3) | Shows error, re-prompts |
| Non-integer input | Clears input stream, re-prompts |
| File not found | Shows error with filename, re-prompts |
| Invalid pattern file (non 0/1 values) | Shows error, re-prompts |
| Incompatible matrix dimensions | Shows A and B dimensions, re-prompts for both files |

---

## Features

- ✅ Works with **any number of processes** (N ≥ 2)
- ✅ Handles **uneven data sizes** automatically via `MPI_Scatterv`
- ✅ **Dynamic load balancing** — Master-Worker with demand-driven chunk dispatch
- ✅ **Non-blocking communication** with computation overlap
- ✅ **Process organization** via `MPI_Comm_split`
- ✅ **Persistent menu** — run multiple algorithms without restarting
- ✅ **Robust error handling** — bad input retries instead of crashing

---

## License

This project was developed for academic purposes at Fayoum University.
