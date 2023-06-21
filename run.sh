#!/bin/bash

make
# srun -p gpu -N 1 ./benchmark 100
# srun -p gpu -N 1 ./benchmark 200
# srun -p gpu -N 1 ./benchmark 500
srun -p gpu -N 1 ./benchmark 1000
srun -p gpu -N 1 ./benchmark 2500
srun -p gpu -N 1 ./benchmark 5000
srun -p gpu -N 1 ./benchmark 7500
srun -p gpu -N 1 ./benchmark 10000