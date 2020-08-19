module Defs

using StaticArrays

export unit
function unit end

export hodge
function hodge end
export ⋆
const ⋆ = hodge

export invhodge
function invhodge end
Base.inv(::typeof(hodge)) = invhodge

export bitsign
bitsign(b::Bool) = b ? -1 : 1
bitsign(i::Integer) = bitsign(isodd(i))

export sort_perm
"""
    sort_perm

Sort and count permutations.
"""
sort_perm(xs::SVector{0}) = xs, 0
sort_perm(xs::SVector{1}) = xs, 0
function sort_perm(xs::SVector{D}) where {D}
    @assert D > 1
    xs1 = xs[StaticArrays.SUnitRange(1, D - 1)]
    xend = xs[end]
    rs1, s1 = sort_perm(xs1)
    i = findfirst(>(xend), rs1)
    i === nothing && (i = D)
    rs = SVector{D}(j < i ? rs1[j] : j == i ? xend : rs1[j - 1] for j in 1:D)
    s = s1 + D - i
    # @assert issorted(rs)
    return rs, s
end

end
