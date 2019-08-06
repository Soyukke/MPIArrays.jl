using MPI, MPIArrays
MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)
x = MPIArray{Int64}(5, 5)
y = MPIArray{Int64}(5, 5)
z = CyclicMPIArray(Int64, 9, 9,  proc_grids=(2, 2), blocksizes=(2, 2))

forlocalpart!(x->fill!(x, rank), z)

# this is error, because z.win is nothing
# show(stdout, "text/plain", z)

rma!(z) do
    if rank == 0
        show(stdout, "text/plain", z)
        println()
    end
end

MPI.Finalize()
