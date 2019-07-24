using MPI, MPIArrays

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
n_proc = MPI.Comm_size(comm)

@assert n_proc == 4

# A = MPIArray{Float64}(8, 8)
# b = MPIArray{Float64}(8)
# forlocalpart!(x -> fill!(x, rank), A)
# forlocalpart!(x -> fill!(x, rank), b)
# sync(A, b)
# B = A*b
# if rank == 0
#     x = localindices(A, 0)
#     println(typeof(x))
#     println("type of B ", typeof(B))
# end
# free(A)
# free(b)

# A = MPIArray{Float64}(8)
# forlocalpart!(x -> fill!(x, rank), A)
# sync(A)
# if rank == 0
#     x = localindices(A, 0)
#     println(typeof(x))
# end
# free(A)


# Cyclic mul! 
# A = MPIArray{Float64}(comm, (2, 2), (2, 2), (8, 8))
# b = MPIArray{Float64}(comm, (4,), (2,), (8,))
# forlocalpart!(x -> fill!(x, rank), A)
# forlocalpart!(x -> fill!(x, rank), b)
# sync(A, b)
# if rank == 0
#     x = localindices(b, 0)
#     println(typeof(x))
#     println(length.(x))
#     len = map(x->length(localindices(b, x-1)), b.partitioning.ranks)
#     println(len)
#     # println("b get $(b[7])")
#     # println("b get $(b[localindices(b, 3)[1]])")
# end
# y = A*b
# if rank == 0
#     println("size $(size(A))")
#     show(stdout, "text/plain", A)
#     println()
#     println("y=A*x size $(size(y))")
#     show(stdout, "text/plain", y)
#     println()
# end
# free(A)
# free(b)


A = MPIArray{Float64}(comm, (2, 2), (2, 2), (9, 7))
# A = MPIArray{Float64}(9, 7)
forlocalpart!(x -> fill!(x, rank), A)
sync(A)
if rank == 0
    println("size $(size(A))")
    show(stdout, "text/plain", A)
    println()
end
free(A)

A = MPIArray{Float64}(comm, (2, 2), (2, 2), (5, 7))
# A = MPIArray{Float64}(9, 7)
forlocalpart!(x -> fill!(x, rank), A)
sync(A)
if rank == 0
    println("size $(size(A))")
    show(stdout, "text/plain", A)
    println()
end
free(A)

A = MPIArray{Float64}(comm, (4, ), (2, ), (16, ))
forlocalpart!(x -> fill!(x, rank), A)
sync(A)
if rank == 0
    println("size $(size(A))")
    show(stdout, "text/plain", A)
    println()
end
free(A)

# useful
A = CyclicMPIArray(Float64, 7, 5)
forlocalpart!(x -> fill!(x, rank), A)
sync(A)
if rank == 0
    println("size $(size(A))")
    show(stdout, "text/plain", A)
    println()
end
free(A)

A = CyclicMPIArray(Float64, 7, 5)
forlocalpart!(x -> fill!(x, rank), A)
sync(A)
if rank == 0
    println("size $(size(A))")
    show(stdout, "text/plain", A)
    println()
end
free(A)





MPI.Finalize()

