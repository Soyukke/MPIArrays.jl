using Test
using MPIArrays

println(methods(CyclicRange))
cr1 = CyclicRange(start=1, stop=11, blocksize=2, n_split=3, split_index=1)
cr2 = CyclicRange(start=1, stop=11, blocksize=2, n_split=3, split_index=2)
cr3 = CyclicRange(start=1, stop=11, blocksize=2, n_split=3, split_index=3)

# a a b b c c a a b b c
@show cr1 cr2 cr3

@show length(cr1) length(cr2) length(cr3)
# nblock = 1, extra_block = 0
@test MPIArrays.number_block_local_min(cr1) == 1
@test collect(cr1) == [1, 2, 7, 8]
@test collect(cr2) == [3, 4, 9, 10]
@test collect(cr3) == [5, 6, 11]
@test length(cr1) == 4
@test length(cr2) == 4
@test length(cr3) == 3

for c in [cr1, cr2, cr3]
for index in 1:length(c)
    @show c[index]
end
end

println("local_index test")
for i in 1:11
    println("global=$i, ", MPIArrays.local_index(cr1, i))
end

# CyclicRangeがAbstractRangeを継承しているのでそのまま使える
println(cr1...)
println(vcat(cr1))
println(collect(cr1))
# show(stdout, cr)


# test 1
cyclic_partitions = MPIArrays.CyclicPartitioning(array_sizes=(9, 9), n_procs=(2, 2), blocksizes=(1, 1))
println(size(cyclic_partitions))
println(cyclic_partitions[1, 1])

println("CyclicPartitioning: local_index test")
show(stdout, "text/plain", map(x -> MPIArrays.local_index(cyclic_partitions, x), Base.Iterators.product(1:9, 1:9)))
println()

# test 2
cyclic_partitions = MPIArrays.CyclicPartitioning(array_sizes=(5, 7), n_procs=(2, 2), blocksizes=(2, 2))
println(size(cyclic_partitions))
println(cyclic_partitions[1, 1])

println("CyclicPartitioning: local_index test")
show(stdout, "text/plain", map(x -> MPIArrays.local_index(cyclic_partitions, x), Base.Iterators.product(1:5, 1:7)))
println()


show(stdout, "text/plain", collect(Base.Iterators.product(cr1, cr1)))
println("")
println("rank 0 local array size: ", length.(cyclic_partitions[1]))
println("rank 1 local array size: ", length.(cyclic_partitions[2]))
println("rank 2 local array size: ", length.(cyclic_partitions[3]))
println("rank 3 local array size: ", length.(cyclic_partitions[4]))