using MPI, MPIArrays

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
n_proc = MPI.Comm_size(comm)

# 1
# default blocksize = (1, 1)
A = CyclicMPIArray(Float64, 7, 5)
rma!(A) do
    forlocalpart!(x -> fill!(x, rank), A)
    sync(A)
    if rank == 0
        println("size $(size(A))")
        show(stdout, "text/plain", A)
        println()
    end
end

# 2
A = CyclicMPIArray(Float64, 9, 7, blocksizes=(2, 2))
rma!(A) do
    forlocalpart!(x -> fill!(x, rank), A)
    sync(A)
    if rank == 0
        println("size $(size(A))")
        show(stdout, "text/plain", A)
        println()
    end
end

# 3
A = CyclicMPIArray(ComplexF64, 7, 5)
rma!(A) do
    forlocalpart!(x -> fill!(x, rank), A)
    sync(A)
    if rank == 0
        println("size $(size(A))")
        show(stdout, "text/plain", A)
        println()
    end
end

# 4
if n_proc % 2 == 0
    A = CyclicMPIArray(Int64, 11, 11, proc_grids=(Int(n_proc/2), 2), blocksizes=(2, 2))
    rma!(A) do
        forlocalpart!(x -> fill!(x, rank), A)
        sync(A)
        if rank == 0
            println("size $(size(A))")
            show(stdout, "text/plain", A)
            println()
        end
    end
end

# 5
if n_proc % 2 == 0
    A = CyclicMPIArray(Int64, 11, 9, proc_grids=(Int(n_proc/2), 2), blocksizes=(3, 3))

    print("rank: $rank, $(size(A.localarray))\n")
    println("$(collect(localindices(A)[1]))")

    rma!(A) do
        forlocalpart!(x -> fill!(x, rank), A)
        sync(A)
        if rank == 0

            println("size $(size(A))")
            show(stdout, "text/plain", A)
            println()
        end
    end
end






MPI.Finalize()

