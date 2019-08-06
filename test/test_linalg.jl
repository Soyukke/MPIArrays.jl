using Test
using MPI, MPIArrays
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
n_proc = MPI.Comm_size(comm)

function test_triu!(k)
    N = 8
    A = CyclicMPIArray(Float64, N, N, proc_grids=(2, 2))
    rma!(A) do
        forlocalpart!(x->fill!(x, rank+1), A)
    end
    B = copy(A)
    rma!(A, B) do
        A_test = convert(Array, A)
        MPI.Barrier(comm)
        triu!(B, k)

        if rank == 0
            println("A")
            show(stdout, "text/plain", A)
            println()
            println("B from A")
            show(stdout, "text/plain", B)
            println()

            println("B localindices")
            show(stdout, "text/plain", B.partitioning)
            println()

            @test convert(Array,B) == triu(A_test, k)
        end
    end
end

function test_triu(k)
    N = 8
    A = CyclicMPIArray(Float64, N, N, proc_grids=(2, 2))
    rma!(A) do
        forlocalpart!(x->fill!(x, rank+1), A)
    end
    B = triu(A, k)
    sync(B)

    rma!(A, B) do
        A_test = convert(Array, A)
        MPI.Barrier(comm)
        if rank == 0
            println("A")
            show(stdout, "text/plain", A)
            println()
            println("B from A")
            show(stdout, "text/plain", B)
            println()

            println("B localindices")
            show(stdout, "text/plain", B.partitioning)
            println()
            @test convert(Array,B) == triu(A_test, k)
        end
    end
end

function test_triu_loop()
    for k in 0:-1:-7
        test_triu!(k)
    end

    for k in 0:-1:-7
        test_triu(k)
    end
end

function test_mul()
    N = 5
    A = CyclicMPIArray(Float64, N, N, proc_grids=(2, 2))
    B = CyclicMPIArray(Float64, N, N, proc_grids=(2, 2))
    rma!(A, B) do
    forlocalpart!(x->fill!(x, rank), A)
    # C = MPIArray{Float64}(N, N)
    # D = C[:, :]
    # C = A * B
    E = convert(Array, A)

    # MPIArray -> Block
    # if rank == 0
        # D = A[:, :]
        # print(typeof(D))
        # show(stdout, "text/plain", getblock(D))
        show(stdout, "text/plain", E)
        println()
    # end
    end
end

test_triu_loop()
test_mul()

MPI.Finalize()