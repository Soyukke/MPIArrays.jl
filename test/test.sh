#!/bin/bash

mpirun -np 4 julia --project=. test/test_cyclicmpiarray.jl
mpirun -np 4 julia --project=. test/test_cyclic_range.jl
mpirun -np 4 julia --project=. test/test_linalg.jl