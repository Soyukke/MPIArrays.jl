export triu, triu!
import LinearAlgebra: triu, triu!
function triu!(A::AbstractMPIArray{T}, k::Integer=0) where T
    zero_ = zero(T)
    forlocalpart!(A) do lA
        gi, gj = localindices(A)
        for (i, gi) in enumerate(gi)
            for (j, gj) in enumerate(gj)
                if gj < gi + k
                    lA[i, j] = zero_
                end
            end
        end
    end
    return A
end

function triu(A::AbstractMPIArray, k::Integer=0)
    B = copy(A)
    return triu!(B, k)
end