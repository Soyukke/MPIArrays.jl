# Cyclic distribute array to each process

For use ScaLAPACK.jl

## Examples

### Example 1

```julia
using MPI, MPIArrays
MPI.Init()

A = CyclicMPIArray(Float64, 7, 5)
forlocalpart!(x -> fill!(x, rank), A)
sync(A)
if rank == 0
    show(stdout, "text/plain", A)
    println()
end

MPI.Finalize()
```

out

```julia
7×5 MPIArray{Float64,2}:
 0.0  0.0  0.0  0.0  0.0
 1.0  1.0  1.0  1.0  1.0
 2.0  2.0  2.0  2.0  2.0
 3.0  3.0  3.0  3.0  3.0
 0.0  0.0  0.0  0.0  0.0
 1.0  1.0  1.0  1.0  1.0
 2.0  2.0  2.0  2.0  2.0
```

### Example 2

blocksize is (2, 2), proccess grid is (4, 1)

```julia
using MPI, MPIArrays
MPI.Init()

A = CyclicMPIArray(Float64, 9, 7, blocksizes=(2, 2))
forlocalpart!(x -> fill!(x, rank), A)
sync(A)
if rank == 0
    show(stdout, "text/plain", A)
    println()
end

MPI.Finalize()
```

out

```
9×7 MPIArray{Float64,2}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0
 2.0  2.0  2.0  2.0  2.0  2.0  2.0
 2.0  2.0  2.0  2.0  2.0  2.0  2.0
 3.0  3.0  3.0  3.0  3.0  3.0  3.0
 3.0  3.0  3.0  3.0  3.0  3.0  3.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
```

### Example 3

blocksize is (2, 2), proccess grid is (2, 2)

```julia
using MPI, MPIArrays
MPI.Init()

A = CyclicMPIArray(Int64, 11, 11, proc_grids=(Int(n_proc/2), 2), blocksizes=(2, 2))
forlocalpart!(x -> fill!(x, rank), A)
sync(A)
if rank == 0
    show(stdout, "text/plain", A)
    println()
end

MPI.Finalize()
```

```
11×11 MPIArray{Int64,2}:
 0  0  2  2  0  0  2  2  0  0  2
 0  0  2  2  0  0  2  2  0  0  2
 1  1  3  3  1  1  3  3  1  1  3
 1  1  3  3  1  1  3  3  1  1  3
 0  0  2  2  0  0  2  2  0  0  2
 0  0  2  2  0  0  2  2  0  0  2
 1  1  3  3  1  1  3  3  1  1  3
 1  1  3  3  1  1  3  3  1  1  3
 0  0  2  2  0  0  2  2  0  0  2
 0  0  2  2  0  0  2  2  0  0  2
 1  1  3  3  1  1  3  3  1  1  3
```

## Work function

- forlocalpart!
- localindices