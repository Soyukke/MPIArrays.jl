using MPI, MPIArrays

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
n_proc = MPI.Comm_size(comm)

# 1
@assert n_proc % 2 == 0
process_grid = (Int(n_proc/2), 2)
A = CyclicMPIArray(Float64, 7, 5, proc_grids=process_grid)
forlocalpart!(x -> fill!(x, rank), A)
sync(A)
if rank == 0
    show(stdout, "text/plain", A)
    println()

    println("pids")
    show(stdout, "text/plain", pids(A))
    println()


    println("blocksizes")
    show(stdout, "text/plain", blocksizes(A))
    println()
end
free(A)

A = CyclicMPIArray(Float64, 7, 9, blocksizes=(2, 2))
forlocalpart!(x -> fill!(x, rank), A)
sync(A)
if rank == 0
    show(stdout, "text/plain", A)
    println()

    println("pids")
    show(stdout, "text/plain", pids(A))
    println()


    println("blocksizes")
    show(stdout, "text/plain", blocksizes(A))
    println()
end
free(A)



MPI.Finalize()