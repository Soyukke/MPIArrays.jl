export CyclicRange

#= make CyclicRange For ScaLAPACK =#

struct CyclicRange{T <: Integer} <: AbstractRange{T}
    start::T # full array's first
    stop::T # full array's last
    blocksize::T # blocksize
    n_split::T # number of process
    split_index::T # what process

    CyclicRange(;start::T, stop::T, blocksize::T, n_split::T, split_index::T) where T <: Integer = new{T}(start, stop, blocksize, n_split, split_index)
end

number_element(r::CyclicRange) = r.stop - r.start + 1

function number_element_local(r::CyclicRange)
    n_element = number_element(r)
    n_block_local_min = number_block_local_min(r)
    n_extra_block = number_block_global_extra(r)

    n_element_local = r.blocksize*n_block_local_min
    n_extra_element = n_element % r.blocksize

    # add extra block
    if r.split_index <= n_extra_block
        n_element_local += r.blocksize
    end
    # add extra element, 
    if r.split_index == n_extra_block + 1 && n_extra_element != 0
        n_element_local += n_extra_element
    end
    return n_element_local
end

function number_block(r::CyclicRange)
    n_element = number_element(r)
    n_block = n_element ÷ r.blocksize
    return n_block
end

function number_block_local_min(r::CyclicRange{T}) where T <: Integer
    n_block = number_block(r)
    n_block_local = n_block ÷ r.n_split
    return n_block_local
end

function number_block_global_extra(r::CyclicRange{T}) where T <: Integer
    n_block = number_block(r)
    n_extra_block = n_block % r.n_split
    return n_extra_block
end

function local_index(r::CyclicRange, i::Integer)
    i_temp = i - 1
    block_position = i_temp ÷ r.blocksize
    local_index = (block_position ÷ r.n_split)*r.blocksize + (i_temp % r.blocksize) + 1
    split_index = block_position % r.n_split + 1
    # mpi rank = split_index - 1, in 0-based local index = extra_element - 1
    return split_index, local_index
end

Base.firstindex(::CyclicRange) = 1

Base.first(r::CyclicRange) = getindex(r, 1)

Base.last(r::CyclicRange) = getindex(r, length(r))

Base.length(r::CyclicRange{T}) where T <: Integer = number_element_local(r)

Base.getindex(r::CyclicRange, i::Integer) = r.blocksize*(r.split_index-1) + r.n_split*r.blocksize*div(i-1, r.blocksize) + mod(i-1, r.blocksize) + 1

Base.show(io::IO, r::CyclicRange) = print(io, "$(r.start):[$(r.split_index) $(r.blocksize) $(r.n_split)]:$(r.stop)")


