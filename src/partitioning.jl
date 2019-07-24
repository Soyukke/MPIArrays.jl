using Base.Iterators: product

abstract type Partitioning{N} <: AbstractArray{Int, N} end



"""
Store the distribution of the array indices over the different partitions.
This class forces a continuous, ordered distribution without overlap

    ContinuousPartitioning(partition_sizes...)

Construct a distribution using the number of elements per partition in each direction, e.g:

```
p = ContinuousPartitioning([2,5,3], [2,3])
```

will construct a distribution containing 6 partitions of 2, 5 and 3 rows and 2 and 3 columns.

"""
struct ContinuousPartitioning{N} <: Partitioning{N}
    ranks::LinearIndices{N,NTuple{N,Base.OneTo{Int}}}
    index_starts::NTuple{N,Vector{Int}}
    index_ends::NTuple{N,Vector{Int}}

    function ContinuousPartitioning(partition_sizes::Vararg{Any,N}) where {N}
        index_starts = Vector{Int}.(undef,length.(partition_sizes))
        index_ends = Vector{Int}.(undef,length.(partition_sizes))
        for (idxstart,idxend,nb_elems_dist) in zip(index_starts,index_ends,partition_sizes)
            currentstart = 1
            currentend = 0
            for i in eachindex(idxstart)
                currentend += nb_elems_dist[i]
                idxstart[i] = currentstart
                idxend[i] = currentend
                currentstart += nb_elems_dist[i]
            end
        end
        ranks = LinearIndices(length.(partition_sizes))
        return new{N}(ranks, index_starts, index_ends)
    end
end

"""
    CyclicPartitioning for ScaLAPACK.jl

```
p = CyclicPartitioning([2,5,3], [2,3], blocksize=(1, 1))
```

will construct a cyclic distribution containing 6 partitions of 2, 5 and 3 rows and 2 and 3 columns.

N is dimension of partition

"""
struct CyclicPartitioning{N} <: Partitioning{N}
    # rank partition mapping
    ranks::LinearIndices{N,NTuple{N,Base.OneTo{Int}}}
    partitions::Array{NTuple{N, CyclicRange}, N}

    function CyclicPartitioning(;array_sizes::NTuple{N, Integer}, n_procs::NTuple{N, Integer}, blocksizes::NTuple{N, Integer}) where {N}
        # process grid array
        ranks = LinearIndices(n_procs)
        # indices array
        partitions = Array{NTuple{N, CyclicRange}, N}(undef, n_procs...)

        for process_indices in product(Base.OneTo.(n_procs)...)
            cyclic_ranges = CyclicRange[]
            for (dim, array_size, n_proc, blocksize) in zip(1:N, array_sizes, n_procs, blocksizes)
                # Index of grid of process in dim dimension
                cyclic_range = CyclicRange(start=1, stop=array_size, blocksize=blocksize, n_split=n_proc, split_index=process_indices[dim])
                push!(cyclic_ranges, cyclic_range)
            end
            partitions[process_indices...] = Tuple(cyclic_ranges)
        end
        # if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        # show(stdout, "text/plain", map(x->length.(x), partitions))
        # end
        return new{N}(ranks, partitions)
    end
end


Base.IndexStyle(::Type{T}) where T <: Partitioning = IndexCartesian()
# number of grid process each dimension
Base.size(p::T) where T <: Partitioning = size(p.ranks)

@inline function Base.getindex(p::ContinuousPartitioning{N}, I::Vararg{Int, N}) where N
    return UnitRange.(getindex.(p.index_starts,I), getindex.(p.index_ends,I))
end

@inline function Base.getindex(p::CyclicPartitioning{N}, I::Vararg{Int, N}) where N
    return p.partitions[I...]
end

function partition_sizes(p::ContinuousPartitioning)
    result = (p.index_ends .- p.index_starts)
    for v in result
        v .+= 1
    end
    return result
end

# Cyclic partion_sizes implemented only 1d and 2d
function partition_sizes(p::CyclicPartitioning{1})
    grids = map(x->length(p.partitions[x][1]), p.ranks[:, 1])
    return (grids, )
end
function partition_sizes(p::CyclicPartitioning{2})
    n_row = map(x->length(p.partitions[x][1]), p.ranks[:, 1])
    n_col = map(x->length(p.partitions[x][2]), p.ranks[1, :])
    result = (n_row, n_col)
    return result
end


"""
  (private method)

Get the rank and local 0-based index
"""
function local_index(p::ContinuousPartitioning, I::NTuple{N,Int}) where {N}
    proc_indices = searchsortedfirst.(p.index_ends, I)
    lininds = LinearIndices(Base.Slice.(p[proc_indices...]))
    return (p.ranks[proc_indices...]-1, lininds[I...] - first(lininds))
end

# 1-dimentional
function local_index(p::CyclicPartitioning{1}, I::NTuple{1,Int})
    cyclic_range = p.partitions[1][1]
    # map global index to local array
    # global_index_mapped = product(cyclic_range...)
    rank_indices = local_index(cyclic_range, I[1])[1]
    local_indices = local_index(cyclic_range, I[1])[2]
    # to 0-based index (rank, indices of local matrix)
    return p.ranks[rank_indices]-1, local_indices-1
end

# 2-dimentional
function local_index(p::CyclicPartitioning{2}, I::NTuple{2,Int})
    # return local indices as 1 dimensional
    rank, (i, j) = local_index_xd(p, I)
    n_local_row, n_local_col = length.(p.partitions[rank+1])
    localindex = i + j*n_local_row
    return rank, localindex
end

function local_index_xd(p::CyclicPartitioning, I::NTuple{N,Int}) where {N}
    # return local indices as Tuple
    # all partitions have same cyclic information
    cyclic_range = p.partitions[1]
    # map global index to local array
    # global_index_mapped = product(cyclic_range...)
    rank_indices = map(x->x[1], local_index.(cyclic_range, I))
    local_indices = map(x->x[2], local_index.(cyclic_range, I))
    # to 0-based index (rank, indices of local matrix)
    return p.ranks[rank_indices...]-1, local_indices .- 1
end

function global_size(p::CyclicPartitioning{1})
    n_index = sum(map(x->length(p.partitions[x]), p.ranks))
    return n_index
end

function global_size(p::CyclicPartitioning{2})
    n_row = sum(map(x->length(p.partitions[x][1]), p.ranks[:, 1]))
    n_col = sum(map(x->length(p.partitions[x][2]), p.ranks[1, :]))
    return (n_row, n_col)
end

blocksizes(p::CyclicPartitioning) = map(x->x.blocksize, p.partitions[1])