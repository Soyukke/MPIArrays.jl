using MPI, MPIArrays

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
n_proc = MPI.Comm_size(comm)

@assert n_proc == 4

# A = MPIArray{Float64}(8, 8)
# forlocalpart!(x -> fill!(x, rank), A)
# sync(A)
# B = A * A
# if rank == 0
#     println("size $(size(A))")
#     show(stdout, "text/plain", A)
#     println("B=A*A size $(size(B))")
#     show(stdout, "text/plain", B)
#     println()
# end
# free(A)
# free(B)


A = MPIArray{Float64}(comm, (2, 2), (2, 2), 8, 8)
# B = A * A
# A = MPIArray{Float64}(9, 7)
forlocalpart!(x -> fill!(x, rank), A)
sync(A)
if rank == 0
    println("size $(size(A))")
    show(stdout, "text/plain", A)
    # println("B=A*A size $(size(B))")
    # show(stdout, "text/plain", B)
    println()
end
free(A)
# free(B)


A = MPIArray{Float64}(comm, (2, 2), (2, 2), 9, 7)
# A = MPIArray{Float64}(9, 7)
forlocalpart!(x -> fill!(x, rank), A)
sync(A)
if rank == 0
    println("size $(size(A))")
    show(stdout, "text/plain", A)
    println()
end
free(A)

A = MPIArray{Float64}(comm, (2, 2), (2, 2), 5, 7)
# A = MPIArray{Float64}(9, 7)
forlocalpart!(x -> fill!(x, rank), A)
sync(A)
if rank == 0
    println("size $(size(A))")
    show(stdout, "text/plain", A)
    println()
end
free(A)

MPI.Finalize()