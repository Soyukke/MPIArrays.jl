using Test
using MPI, MPIArrays
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
n_proc = MPI.Comm_size(comm)

function test_triu!(k)
    N = 8
    A = CyclicMPIArray(Float64, N, N, proc_grids=(2, 2))
    forlocalpart!(x->fill!(x, rank+1), A)
    sync(A)
    A_test = convert(Array, A)
    B = copy(A)
    triu!(B, k)
    sync(B)

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
    free(A, B)
end

function test_triu(k)
    N = 8
    A = CyclicMPIArray(Float64, N, N, proc_grids=(2, 2))
    forlocalpart!(x->fill!(x, rank+1), A)
    sync(A)
    A_test = convert(Array, A)
    B = triu(A, k)
    sync(B)

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
    free(A, B)
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
    forlocalpart!(x->fill!(x, rank), A)
    sync(A)
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

test_mul()

MPI.Finalize()