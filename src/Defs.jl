module Defs

using StaticArrays

export unit
function unit end

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

function sort_perm(xs::AbstractVector)
    D = length(xs)
    D <= 1 && return xs, 0
    xs1 = @view xs[1:(D - 1)]
    xend = xs[end]
    rs1, s1 = sort_perm(xs1)
    i = findfirst(>(xend), rs1)
    i === nothing && (i = D)
    rs = [j < i ? rs1[j] : j == i ? xend : rs1[j - 1] for j in 1:D]
    s = s1 + D - i
    # @assert issorted(rs)
    return rs, s
end

# Using mergesort, which is probably overkill and slower
# sort_perm(xs::SVector{0}) = xs, false
# sort_perm(xs::SVector{1}) = xs, false
# function sort_perm(xs::SVector{2})
#     if xs[1] <= xs[2]
#         xs, false
#     else
#         reverse(xs), true
#     end
# end
# function sort_perm(xs::SVector{D}) where D
#     @assert D >= 2
#     D1 = prevpow(2, D)
#     @assert 1 <= D1 < D
#     D2 = D - D1
#     @assert 1 <= D2
#     xs1 = xs[StaticArrays.SUnitRange(1, D1)]
#     xs2 = xs[StaticArrays.SUnitRange(D1+1, D2)]
#     rs1, s1 = sort_perm(xs1)
#     rs2, s2 = sort_perm(xs2)
#     i1 = 1
#     i2 = 1
#     rs = xs                     # should be undefined
#     for i in 1:D
#         if rs1[i1] < rs2[i2]
#             rs = setindex(rs, rs1[i1], i)
#             i1 += 1
#             if i1 == D
#                 for j in i+1:D
#                     rs = setindex(rs, rs2[i2], j)
#                     i2 += 1
#                 end
#                 break
#             end
#         else
#             rs = setindex(rs, rs2[i2], i)
#             i2 += 1
#             if i2 == D
#                 for j in i+1:D
#                     rs = setindex(rs, rs1[i1], j)
#                     i1 += 1
#                 end
#                 break
#             end
#         end
#     end
#     
#     @assert i1 == D1
#     @assert i2 == D2
#     @assert i == D
#     @assert issorted(rs)
#     rs, s
# end

end
