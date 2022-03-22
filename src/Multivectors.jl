module Multivectors

using ComputedFieldTypes
using LinearAlgebra
using Random
using StaticArrays

using ..Defs
using ..Forms

export Multivector
@computed struct Multivector{D,M,T}
    elts::SVector{numelts(Val(D), Val(M)),T}
end

@generated function numelts(::Val{D}, ::Val{M}) where {D,M}
    D::Int
    @assert D >= 0
    M::Unsigned
    @assert count_ones(~zero(M)) >= D
    return count_ones(M)
end

# Constructor without explicit type
Multivector{D,M}(elts::SVector{N,T}) where {D,M,N,T} = Multivector{D,M,T}(elts)

# Constructor with added computed type (which must match)
function Multivector{D,M,T,X}(args...) where {D,M,T,X}
    return Multivector{D,M,T}(args...)::Multivector{D,M,T,X}
end

const IteratorTypes = Union{Base.Generator,Iterators.Flatten}
function Multivector{D,M,T}(gen::IteratorTypes) where {D,M,T}
    N = length(Multivector{D,M})
    return Multivector{D,M,T}(SVector{N,T}(elts))
end
function Multivector{D,M}(gen::IteratorTypes) where {D,M}
    @assert IteratorEltype(typeof(gen)) == HasEltype()
    N = length(Multivector{D,M})
    T = eltype(gen)
    return Multivector{D,M,T}(SVector{N,T}(elts))
end

function Multivector{D,M,T}(tup::Tuple) where {D,M,T}
    N = length(Multivector{D,M})
    return Multivector{D,M,T}(SVector{N,T}(tup))
end
function Multivector{D,M}(tup::Tuple) where {D,M}
    N = length(Multivector{D,M})
    return Multivector{D,M}(SVector{N}(tup))
end
function Multivector{D,M}(tup::Tuple{}) where {D,M}
    return error("Cannot create Multivector from emtpy tuple without specifying type")
end

# Conversions
function Multivector{D,M,T}(f::Multivector{D,M}) where {D,M,T}
    N = length(Multivector{D,M})
    return Multivector{D,M,T}(SVector{N,T}(f.elts))
end

################################################################################

# I/O

function Base.show(io::IO, x::Multivector{D,M,T}) where {D,M,T}
    print(io, "$T⟦")
    for n in 1:length(x)
        n > 1 && print(io, ",")
        print(io, x[n])
    end
    print(io, "⟧{D=$D,M=0b$(string(M; base = 2))}")
    return nothing
end

################################################################################

# Iterate over masks

setbit(bits::Unsigned, n) = bits | (one(bits) << n)
getbit(bits::Unsigned, n) = (bits & (1 << n)) != 0
nbits(D) = setbit(zero(UInt64), D)

struct BinaryTerm{I1,I2}
    i1::I1
    i2::I2
end
@generated function binary_algorithm(::Val{D}, ::Val{M1}, ::Val{M2}) where {D,M1,M2}
    D::Int
    @assert D >= 0
    M1::Unsigned
    @assert M1 < nbits(nbits(D))
    M2::Unsigned
    @assert M2 < nbits(nbits(D))
    M = M1 | M2
    r = BinaryTerm[]
    i1 = i2 = i = 0
    for n in 0:(nbits(D) - 1)
        i1 += getbit(M1, n)
        i2 += getbit(M2, n)
        i += getbit(M, n)
        if getbit(M, n)
            i1′ = getbit(M1, n) ? i1 : nothing
            i2′ = getbit(M2, n) ? i2 : nothing
            push!(r, BinaryTerm(i1′, i2′))
        end
    end
    @assert i1 == count_ones(M1)
    @assert i2 == count_ones(M2)
    @assert i == count_ones(M)
    return Tuple(r)
end

################################################################################

# Comparisons

@inline @inbounds function eval_eq_term(term::BinaryTerm, x1, x2)
    x1i1 = term.i1 === nothing ? zero(eltype(x1)) : x1[term.i1]
    x2i2 = term.i2 === nothing ? zero(eltype(x2)) : x2[term.i2]
    return x1i1 == x2i2
end
function Base.:(==)(x1::Multivector{D,M1,T1}, x2::Multivector{D,M2,T2}) where {D,M1,M2,T1,T2}
    algorithm = binary_algorithm(Val(D), Val(M1), Val(M2))
    # return all(map(term -> eval_eq_term(term, x1, x2), algorithm))
    # We want left-to-right evaluation without shortcuts to enable
    # SIMD vectorization
    return mapfoldl(term -> eval_eq_term(term, x1, x2), &, algorithm; init=true)
end
# function Base.:(==)(x1::Multivector{D}, x2::Multivector{D}) where {D}
#     return all(iszero, x1 - x2)
# end
function Base.isequal(x1::Multivector{D,M}, x2::Multivector{D,M}) where {D,M}
    return isequal(x1.elts, x2.elts)
end
Base.hash(x1::Multivector, h::UInt) = hash(hash(x1.elts, h), UInt(0x60671433))

################################################################################

# Multivectors are collections

Base.eltype(::Type{<:Multivector{D,M,T}}) where {D,M,T} = T
Base.firstindex(::Type{<:Multivector}) = 1
Base.firstindex(x::Multivector) = firstindex(typeof(x))
Base.iterate(x::Multivector, state...) = iterate(x.elts, state...)
function Base.lastindex(::Type{<:Multivector{D,M}}) where {D,M}
    return length(Multivector{D,M})
end
Base.lastindex(x::Multivector) = lastindex(typeof(x))
Base.length(::Type{<:Multivector{D,M}}) where {D,M} = count_ones(M)
Base.length(x::Multivector) = length(typeof(x))

Base.getindex(x::Multivector, ind::Integer) = x.elts[ind]
function Base.setindex(x::Multivector{D,M}, val, ind::Integer) where {D,M}
    return Multivector{D,M}(Base.setindex(x.elts, val, ind))
end

function Base.map(f, x::Multivector{D,M}, ys::Multivector{D,M}...) where {D,M}
    return Multivector{D,M}(map(f, x.elts, map(y -> y.elts, ys)...))
end
function Base.reduce(f, x::Multivector{D,M}, ys::Multivector{D,M}...; kws...) where {D,M}
    return reduce(f, x.elts, map(y -> y.elts, ys)...; kws...)
end
function Base.mapreduce(f, op, x::Multivector{D,M}, ys::Multivector{D,M}...; kws...) where {D,M}
    return mapreduce(f, op, x.elts, map(y -> y.elts, ys)...; kws...)
end

################################################################################

# Multivectors form a vector space

function Base.rand(rng::AbstractRNG, ::Random.SamplerType{<:Multivector{D,M,T}}) where {D,M,T}
    N = length(Multivector{D,M})
    return Multivector{D,M}(rand(rng, SVector{N,T}))
end
function Base.zero(::Type{<:Multivector{D,M,T}}) where {D,M,T}
    N = length(Multivector{D,M})
    return Multivector{D,M}(zero(SVector{N,T}))
end
function Base.zero(::Type{<:Multivector{D,M}}) where {D,M}
    return zero(Multivector{D,M,Float64})
end
Base.zero(::Type{<:Multivector{D}}) where {D} = zero(Multivector{D,0})
Base.zero(x::Multivector) = zero(typeof(x))

function Defs.unit(::Type{<:Multivector{D,M,T}}, inds::SVector{N,<:Integer}) where {D,M,T,N}
    M::Unsigned
    n = lst2lin(Val(D), Val(M), inds)
    @assert n != 0
    return Multivector{D,M}((one(T),))
end
function Defs.unit(::Type{<:Multivector{D,M}}, inds::SVector{N,<:Integer}) where {D,M,N}
    return unit(Multivector{D,M,Float64}, inds)
end
Defs.unit(F::Type{<:Multivector}, inds::Tuple{}) = unit(F, SVector{0,Int}())
Defs.unit(F::Type{<:Multivector}, inds::Tuple) = unit(F, SVector(inds))
Defs.unit(F::Type{<:Multivector}, inds::Integer...) = unit(F, inds)

Base.:+(x::Multivector{D,M}) where {D,M} = Multivector{D,M}(+x.elts)
Base.:-(x::Multivector{D,M}) where {D,M} = Multivector{D,M}(-x.elts)

@inline @inbounds function eval_add_term(term::BinaryTerm, x1, x2)
    term.i1 === nothing && return x2[term.i2]
    term.i2 === nothing && return x1[term.i1]
    return x1[term.i1] + x2[term.i2]
end
function Base.:+(x1::Multivector{D,M1}, x2::Multivector{D,M2}) where {D,M1,M2}
    M = M1 | M2
    M == 0 && return Multivector{D,M,typeof(zero(eltype(x1)) + zero(eltype(x2)))}(())
    algorithm = binary_algorithm(Val(D), Val(M1), Val(M2))
    return Multivector{D,M}(map(term -> eval_add_term(term, x1, x2), algorithm))
end
@inline @inbounds function eval_sub_term(term::BinaryTerm, x1, x2)
    term.i1 === nothing && return -x2[term.i2]
    term.i2 === nothing && return x1[term.i1]
    return x1[term.i1] - x2[term.i2]
end
function Base.:-(x1::Multivector{D,M1}, x2::Multivector{D,M2}) where {D,M1,M2}
    M = M1 | M2
    M == 0 && return Multivector{D,M,typeof(zero(eltype(x1)) - zero(eltype(x2)))}(())
    algorithm = binary_algorithm(Val(D), Val(M1), Val(M2))
    return Multivector{D,M}(map(term -> eval_sub_term(term, x1, x2), algorithm))
end

Base.:*(x::Multivector{D,M}, a) where {D,M} = Multivector{D,M}(x.elts * a)
Base.:/(x::Multivector{D,M}, a) where {D,M} = Multivector{D,M}(x.elts / a)
Base.:*(a, x::Multivector{D,M}) where {D,M} = Multivector{D,M}(a * x.elts)
Base.:\(a, x::Multivector{D,M}) where {D,M} = Multivector{D,M}(a \ x.elts)

################################################################################

# Internal representation

const bits2uint = Forms.bits2uint
const uint2bits = Forms.uint2bits

"""
all bit indices for dimension `D` with stored elements `M`
"""
function dim_bitindices_slow(D::Int, M::Unsigned)
    bitindices = Vector{SVector{D,Bool}}()
    for ibits in CartesianIndex(ntuple(d -> 0, D)):CartesianIndex(ntuple(d -> 1, D))
        bits = SVector{D,Bool}(Tuple(ibits))
        if getbit(M, bits2uint(bits))
            push!(bitindices, bits)
        end
    end
    return bitindices
end
@generated function dim_bitindices(::Val{D}, ::Val{M}) where {D,M}
    return dim_bitindices_slow(D, M)
end

"""
all linear indices for dimension `D` with stored elements `M`
"""
function dim_linindices_slow(D::Int, M::Unsigned)
    bitindices = dim_bitindices(Val(D), Val(M))
    linindices = zeros(Int, nbits(D))
    for (lin, bits) in enumerate(bitindices)
        ubits = bits2uint(bits)
        linindices[ubits + 1] = lin
    end
    return linindices
end
@generated function dim_linindices(::Val{D}, ::Val{M}) where {D,M}
    return dim_linindices_slow(D, M)
end

"""
convert a linear index to a bit index
"""
function lin2bit(::Val{D}, ::Val{M}, lin::Int) where {D,M}
    D::Int
    M::Unsigned
    return dim_bitindices(Val(D), Val(M))[lin]::SVector{D,Bool}
end

"""
Convert a bit index to a linear index. Return 0 for failure.
"""
function bit2lin(::Val{D}, ::Val{M}, bits::SVector{D,Bool}) where {D,M}
    D::Int
    M::Unsigned
    return dim_linindices(Val(D), Val(M))[bits2uint(bits) + 1]::Int
end

"""
Convert a bit index to an index list
"""
function bit2lst(::Val{D}, bits::SVector{D,Bool}) where {D}
    D::Int
    list = Int[]
    for d in 1:D
        if bits[d]
            push!(list, d)
        end
    end
    return list::Vector{Int}
end

"""
Convert an index list to a bit index
"""
function lst2bit(::Val{D}, list::AbstractVector{<:Integer}) where {D}
    D::Int
    @assert issorted(list)
    bits = zero(SVector{D,Bool})
    for ind in list
        bits = Base.setindex(bits, true, ind)
    end
    return bits::SVector{D,Bool}
end

function lin2lst(::Val{D}, ::Val{M}, lin::Int) where {D,M}
    return bit2lst(Val(D), lin2bit(Val(D), Val(M), lin))
end

function lst2lin(::Val{D}, ::Val{M}, list::AbstractVector{<:Integer}) where {D,M}
    return bit2lin(Val(D), Val(M), lst2bit(Val(D), list))
end

################################################################################

# Multivector algebra

# one

@inline function Base.one(::Type{<:Multivector{D,M,T}}) where {D,M,T}
    return unit(Multivector{D,M,T})
end
@inline function Base.one(::Type{<:Multivector{D,M}}) where {D,M}
    return one(Multivector{D,M,Float64})
end
@inline function Base.one(::Type{<:Multivector{D}}) where {D}
    return one(Multivector{D,UInt64(1)})
end
@inline Base.one(x::Multivector) = one(typeof(x))

# reverse (~)

@generated function reverse_algorithm(::Val{D}, ::Val{M1}) where {D,M1}
    res = Bool[]
    N1 = numelts(Val(D), Val(M1))
    for n1 in 1:N1
        bits1 = lin2bit(Val(D), Val(M1), n1)
        R1 = count(bits1)
        s = isodd((R1 - 1) * R1 ÷ 2)
        push!(res, s)
    end
    return SVector{length(res),Bool}(res)
end
function Base.reverse(x::Multivector{D,M}) where {D,M}
    M == 0 && return x
    algorithm = reverse_algorithm(Val(D), Val(M))
    return Multivector{D,M}(map((s, x) -> s ? -x : x, algorithm, x.elts))
end
@inline Base.:~(x::Multivector) = reverse(x)

# hodge (⋆, \\star)

@generated function hodge_mask(::Val{D}, ::Val{M1}) where {D,M1}
    N1 = numelts(Val(D), Val(M1))
    M = zero(M1)
    for n1 in 1:N1
        bits1 = lin2bit(Val(D), Val(M1), n1)
        bitsr = .~bits1
        M = setbit(M, bits2uint(bitsr))
    end
    return M
end
@generated function hodge_algorithm(::Val{D}, ::Val{M1}) where {D,M1}
    N1 = numelts(Val(D), Val(M1))
    M = hodge_mask(Val(D), Val(M1))
    N = numelts(Val(D), Val(M))
    elts = Any[nothing for n in 1:N]
    for n1 in 1:N1
        bits1 = lin2bit(Val(D), Val(M1), n1)
        bitsr = .~bits1
        _, parity = sort_perm([bit2lst(Val(D), bits1)
                               bit2lst(Val(D), bitsr)])
        s = isodd(parity)
        ind = bit2lin(Val(D), Val(M), bitsr)
        elts[ind] = (s, n1)
    end
    @assert !any(==(nothing), elts)
    return SVector{N,Tuple{Bool,Int}}(elts)
end
@inline @inbounds function eval_hodge_term(term::Tuple{Bool,Int}, x1)
    s, i = term
    return bitsign(s) * x1[i]
end
function Forms.hodge(x1::Multivector{D,M1}) where {D,M1}
    @assert 0 <= D
    M1::Unsigned
    M = hodge_mask(Val(D), Val(M1))
    M == 0 && return zero(Multivector{D,M,eltype(x1)})
    algorithm = hodge_algorithm(Val(D), Val(M1))::SVector
    return Multivector{D,M}(map(term -> eval_hodge_term(term, x1), algorithm))
end

# invhodge (inv(⋆), inv(\\star))

@generated function invhodge_mask(::Val{D}, ::Val{M1}) where {D,M1}
    N1 = numelts(Val(D), Val(M1))
    M = zero(M1)
    for n1 in 1:N1
        bits1 = lin2bit(Val(D), Val(M1), n1)
        bitsr = .~bits1
        M = setbit(M, bits2uint(bitsr))
    end
    return M
end
@generated function invhodge_algorithm(::Val{D}, ::Val{M1}) where {D,M1}
    N1 = numelts(Val(D), Val(M1))
    M = invhodge_mask(Val(D), Val(M1))
    N = numelts(Val(D), Val(M))
    elts = Any[nothing for n in 1:N]
    for n1 in 1:N1
        bits1 = lin2bit(Val(D), Val(M1), n1)
        bitsr = .~bits1
        # The "opposite" parity as `hodge`
        _, parity = sort_perm([bit2lst(Val(D), bitsr)
                               bit2lst(Val(D), bits1)])
        s = isodd(parity)
        ind = bit2lin(Val(D), Val(M), bitsr)
        elts[ind] = (s, n1)
    end
    @assert !any(==(nothing), elts)
    return SVector{N,Tuple{Bool,Int}}(elts)
end
@inline @inbounds function eval_invhodge_term(term::Tuple{Bool,Int}, x1)
    s, i = term
    return bitsign(s) * x1[i]
end
function Forms.invhodge(x1::Multivector{D,M1}) where {D,M1}
    @assert 0 <= D
    M1::Unsigned
    M = invhodge_mask(Val(D), Val(M1))
    M == 0 && return zero(Multivector{D,M,eltype(x1)})
    algorithm = invhodge_algorithm(Val(D), Val(M1))::SVector
    return Multivector{D,M}(map(term -> eval_invhodge_term(term, x1), algorithm))
end

# conj

@inline function Base.conj(x::Multivector{D,M}) where {D,M}
    return Multivector{D,M}(conj.(x.elts))
end

# wedge (∧, \\wedge)

struct WedgeTerm
    sign::Bool
    n1::Int
    n2::Int
end
struct WedgeElt{N}
    terms::NTuple{N,WedgeTerm}
end
@generated function wedge_mask(::Val{D}, ::Val{M1}, ::Val{M2}) where {D,M1,M2}
    D::Int
    M1::Unsigned
    M2::Unsigned
    N1 = numelts(Val(D), Val(M1))
    N2 = numelts(Val(D), Val(M2))
    M = zero(typeof(M1 | M2))
    for n1 in 1:N1, n2 in 1:N2
        bits1 = lin2bit(Val(D), Val(M1), n1)
        bits2 = lin2bit(Val(D), Val(M2), n2)
        if !any(bits1 .& bits2)
            bitsr = bits1 .| bits2
            M = setbit(M, bits2uint(bitsr))
        end
    end
    return M
end
@generated function wedge_algorithm(::Val{D}, ::Val{M1}, ::Val{M2}) where {D,M1,M2}
    D::Int
    M1::Unsigned
    M2::Unsigned
    M = wedge_mask(Val(D), Val(M1), Val(M2))
    N1 = numelts(Val(D), Val(M1))
    N2 = numelts(Val(D), Val(M2))
    N = numelts(Val(D), Val(M))
    elts = [WedgeTerm[] for n in 1:N]
    for n1 in 1:N1, n2 in 1:N2
        bits1 = lin2bit(Val(D), Val(M1), n1)
        bits2 = lin2bit(Val(D), Val(M2), n2)
        if !any(bits1 .& bits2)
            bitsr = bits1 .| bits2
            @assert getbit(M, bits2uint(bitsr))
            _, parity = sort_perm([bit2lst(Val(D), bits1)
                                   bit2lst(Val(D), bits2)])
            s = isodd(parity)
            ind = bit2lin(Val(D), Val(M), bitsr)
            push!(elts[ind], WedgeTerm(s, n1, n2))
        end
    end
    elts = map(elt -> WedgeElt{length(elt)}(Tuple(elt)), elts)
    return Tuple(elts)
end
@inline @inbounds function eval_wedge_elt(elt::WedgeElt{N}, x1, x2) where {N}
    U = typeof(one(eltype(x1)) * one(eltype(x2)))
    N == 0 && return zero(U)
    term = elt.terms[1]
    r = bitsign(term.sign) * x1[term.n1] * x2[term.n2]
    for n in 2:N
        term = elt.terms[n]
        r += bitsign(term.sign) * x1[term.n1] * x2[term.n2]
    end
    return r
end
function Forms.wedge(x1::Multivector{D,M1}, x2::Multivector{D,M2}) where {D,M1,M2}
    @assert 0 <= D
    M1::Unsigned
    M2::Unsigned
    M = wedge_mask(Val(D), Val(M1), Val(M2))
    M == 0 && return zero(Multivector{D,M,typeof(one(eltype(x1)) * one(eltype(x2)))})
    algorithm = wedge_algorithm(Val(D), Val(M1), Val(M2))::Tuple
    return Multivector{D,M}(map(elt -> eval_wedge_elt(elt, x1, x2), algorithm))
end
@inline Forms.wedge(x::Multivector) = x
@inline function Forms.wedge(x1::Multivector, x2::Multivector, x3s::Multivector...)
    return wedge(wedge(x1, x2), x3s...)
end
@inline function Forms.wedge(xs::SVector{0,<:Multivector{D,M,T}}) where {D,M,T}
    return one(Multivector{D,one(M),T})
end
@inline Forms.wedge(xs::SVector{1,<:Multivector}) = xs[1]
@inline function Forms.wedge(xs::SVector{N,<:Multivector}) where {N}
    return ∧(pop(xs)) ∧ last(xs)
end

# vee (∨, \\vee)

@inline Forms.vee(x1::Multivector, x2::Multivector) = inv(⋆)(⋆x1 ∧ ⋆x2)
@inline Forms.vee(x::Multivector) = x
@inline vee′(x1::Multivector) = x1
@inline vee′(x1::Multivector, x2::Multivector) = x1 ∧ ⋆x2
@inline function vee′(x1::Multivector, x2::Multivector, x3s::Multivector...)
    return vee′(x1 ∧ ⋆x2, x3s...)
end
@inline function Forms.vee(x1::Multivector, x2::Multivector, x3s::Multivector...)
    # vee(x1 ∨ x2, x3s...)
    return inv(⋆)(vee′(⋆x1, x2, x3s...))
end
@inline function Forms.vee(xs::SVector{0,<:Multivector{D,M,T}}) where {D,M,T}
    return ⋆one(Multivector{D,one(M),T})
end
@inline Forms.vee(xs::SVector{1,<:Multivector}) = xs[1]
@inline function vee′(xs::SVector{0,<:Multivector{D,M,T}}) where {D,M,T}
    return one(Multivector{D,one(M),T})
end
@inline vee′(xs::SVector{1,<:Multivector}) = ⋆xs[1]
@inline vee′(xs::SVector{N,<:Multivector}) where {N} = vee′(pop(xs)) ∧ ⋆last(xs)
@inline function Forms.vee(xs::SVector{N,<:Multivector}) where {N}
    # vee(pop(xs)) ∨ last(xs)
    return inv(⋆)(vee′(xs))
end

# dot (⋅, \\cdot)

@inline LinearAlgebra.dot(x1::Multivector, x2::Multivector) = x1 ∨ ⋆x2

@inline function Forms.norm2(x::Multivector{D,M}) where {D,M}
    M == 0 && return zero(eltype(x))
    return (x ⋅ x)[1]
end
@inline LinearAlgebra.norm(x::Multivector) = sqrt(norm2(x))

# cross (×, \\times)

@inline LinearAlgebra.cross(x1::Multivector, x2::Multivector) = ⋆(x1 ∧ x2)

# geometric product (*)

struct MulTerm
    sign::Bool
    n1::Int
    n2::Int
end
struct MulElt{N}
    terms::NTuple{N,MulTerm}
end
@generated function mul_mask(::Val{D}, ::Val{M1}, ::Val{M2}) where {D,M1,M2}
    D::Int
    M1::Unsigned
    M2::Unsigned
    N1 = numelts(Val(D), Val(M1))
    N2 = numelts(Val(D), Val(M2))
    M = zero(typeof(M1 | M2))
    for n1 in 1:N1, n2 in 1:N2
        bits1 = lin2bit(Val(D), Val(M1), n1)
        bits2 = lin2bit(Val(D), Val(M2), n2)
        bitsr = (bits1 .| bits2) .& .~(bits1 .& bits2)
        M = setbit(M, bits2uint(bitsr))
    end
    return M
end
@generated function mul_algorithm(::Val{D}, ::Val{M1}, ::Val{M2}) where {D,M1,M2}
    D::Int
    M1::Unsigned
    M2::Unsigned
    M = mul_mask(Val(D), Val(M1), Val(M2))
    N1 = numelts(Val(D), Val(M1))
    N2 = numelts(Val(D), Val(M2))
    N = numelts(Val(D), Val(M))
    elts = [MulTerm[] for n in 1:N]
    for n1 in 1:N1, n2 in 1:N2
        bits1 = lin2bit(Val(D), Val(M1), n1)
        bits2 = lin2bit(Val(D), Val(M2), n2)
        bitsr = (bits1 .| bits2) .& .~(bits1 .& bits2)
        @assert getbit(M, bits2uint(bitsr))
        _, parity = sort_perm([bit2lst(Val(D), bits1)
                               bit2lst(Val(D), bits2)])
        s = isodd(parity)
        ind = bit2lin(Val(D), Val(M), bitsr)
        push!(elts[ind], MulTerm(s, n1, n2))
    end
    elts = map(elt -> MulElt{length(elt)}(Tuple(elt)), elts)
    return Tuple(elts)
end
@inline @inbounds function eval_mul_elt(elt::MulElt{N}, x1, x2) where {N}
    U = typeof(one(eltype(x1)) * one(eltype(x2)))
    N == 0 && return zero(U)
    term = elt.terms[1]
    r = bitsign(term.sign) * x1[term.n1] * x2[term.n2]
    for n in 2:N
        term = elt.terms[n]
        r += bitsign(term.sign) * x1[term.n1] * x2[term.n2]
    end
    return r
end
function Base.:*(x1::Multivector{D,M1}, x2::Multivector{D,M2}) where {D,M1,M2}
    @assert 0 <= D
    M1::Unsigned
    M2::Unsigned
    M = mul_mask(Val(D), Val(M1), Val(M2))
    M == 0 && return zero(Multivector{D,M,typeof(one(eltype(x1)) * one(eltype(x2)))})
    algorithm = mul_algorithm(Val(D), Val(M1), Val(M2))::Tuple
    return Multivector{D,M}(map(elt -> eval_mul_elt(elt, x1, x2), algorithm))
end
@inline Base.:*(x::Multivector) = x
@inline function Base.:*(x1::Multivector, x2::Multivector, x3s::Multivector...)
    return *(x1 * x2, x3s...)
end
@inline function Base.:*(xs::SVector{0,<:Multivector{D,M,T}}) where {D,M,T}
    return one(Multivector{D,one(M),T})
end
@inline Base.:*(xs::SVector{1,<:Multivector}) = xs[1]
@inline Base.:*(xs::SVector{N,<:Multivector}) where {N} = *(pop(xs)) * last(xs)

end
