module StaticTensors

using ComputedFieldTypes
using Random
using StaticArrays

"""
    TensorSymmetry{D,R,N}

    `D`: dimension
    `R`: rank
    `N`: stored elts
"""
@computed struct TensorSymmetry{D,R,N}
    stored_elts::SVector{N,SVector{R,Int}}
    computed_elts::SArray{Tuple{ntuple(d -> D, r)...},Tuple{Int,Int}} # (stored element, factor)
end
Base.length(::TensorSymmetry{D,R,N}) where {D,R,N} = N
Base.length(::Val{D}, ::Val{R}, ::TensorSymmetry{D,R,N}) where {D,R,N} = N
tdim(::TensorSymmetry{D,R,N}) where {D,R,N} = D
trank(::TensorSymmetry{D,R,N}) where {D,R,N} = R

export STensor
@computed struct STensor{D,R,TS,T}
    elts::SVector{length(Val(D), Val(R), TS),T}
end

################################################################################

# STensors are collections

Base.eltype(::Type{<:STensor{D,R,T}}) where {D,R,T} = T
Base.firstindex(::Type{<:STensor}) = 1
Base.firstindex(x::STensor) = firstindex(typeof(x))
Base.iterate(x::STensor, state...) = iterate(x.elts, state...)
Base.ndims(::Type{<:STensor{D,R}}) where {D,R} = R
Base.ndims(x::STensor) = ndims(typeof(x))
Base.lastindex(::Type{<:STensor{D,R}}) where {D,R} = length(STensor{D,R})
Base.lastindex(x::STensor) = lastindex(typeof(x))
Base.length(::Type{<:STensor{D,R,TS}}) where {D,R,TS} = length(TS)
Base.length(x::STensor) = length(typeof(x))
Base.size(::Type{<:STensor{D,R}}) where {D,R} = ntuple(d -> D, R)
Base.size(x::STensor) = size(typeof(x))
Base.size(::Type{<:STensor{D}}, r) where {D} = D
Base.size(x::STensor, r) = size(typeof(x))

Base.getindex(x::STensor, ind::Integer) = x.elts[ind]
function Base.getindex(x::STensor{D,R,TS,T}, inds::SVector{R}) where {D,R,TS,T}
    sf = TS.computed_elts[inds]
    sf[2] == 0 && return zero(T)
    return sf[2] * x.elts[sf[1]]
end
Base.getindex(x::STensor, inds::Tuple{}) = x[SVector{0,Int}()]
Base.getindex(x::STensor, inds::Tuple) = x[SVector(inds)]
Base.getindex(x::STensor, inds::Integer...) = x[inds]

function Base.setindex(x::STensor{D,R,TS}, val, ind::Integer) where {D,R,TS}
    return STensor{D,R,TS}(Base.setindex(x.elts, val, ind))
end
function Base.setindex(x::STensor{D,R,TS}, val, inds::SVector{R}) where {D,R,TS}
    sf = TS.computed_elts[inds]
    @assert sf[2] == 1
    return Base.setindex(x, val, sf[1])
end
function Base.setindex(x::STensor, val, inds::Tuple{})
    return Base.setindex(x, val, SVector{0,Int}())
end
Base.setindex(x::STensor, val, inds::Tuple) = Base.setindex(x, val, SVector(inds))
Base.setindex(x::STensor, val, inds::Integer...) = Base.setindex(x, val, inds)

function Base.map(f, x::STensor{D,R,TS}, ys::STensor{D,R,TS}...) where {D,R,TS}
    return STensor{D,R,TS}(map(f, x.elts, map(y -> y.elts, ys)...))
end
function Base.reduce(f, x::STensor{D,R,TS}, ys::STensor{D,R,TS}...) where {D,R,TS}
    return reduce(f, x.elts, map(y -> y.elts, ys)...)
end

################################################################################

# STensors form a vector space

function Base.rand(rng::AbstractRNG, ::Random.SamplerType{<:STensor{D,R,TS,T}}) where {D,R,TS,T}
    N = length(TS)
    return STensor{D,R,TS}(rand(rng, SVector{N,T}))
end
function Base.zero(::Type{<:STensor{D,R,TS,T}}) where {D,R,TS,T}
    N = length(TS)
    return STensor{D,R,TS}(zero(SVector{N,T}))
end
Base.zero(::Type{<:STensor{D,R,TS}}) where {D,R,TS} = zero(STensor{D,R,TS,Float64})
Base.zero(x::STensor) = zero(typeof(x))
Base.iszero(x::STensor) = all(iszero, x.elts)

# Deprecated
# @inline Base.zeros(::Type{<:STensor{D,R,TS,T}}) where {D,R,TS,T} = zero(STensor{D,R,TS,T})
# @inline Base.zeros(::Type{<:STensor{D,R}}) where {D,R} = zeros(STensor{D,R,Float64})
# @inline Base.zeros(x::STensor) = zeros(typeof(x))

Defs.unit(::Type{<:STensor{D,R,TS,T}}, ind::Integer) where {D,R,TS,T} = setindex(zero(STensor{D,R,TS,T}), one(T), ind)
Defs.unit(::Type{<:STensor{D,R,TS}}, ind::Integer) where {D,R,TS} = unit(STensor{D,R,TS,Float64}, ind)
Defs.unit(::Type{<:STensor{D,R,TS,T}}, inds::SVector{R}) where {D,R,TS,T} = setindex(zero(STensor{D,R,TS,T}), one(T), inds)
Defs.unit(::Type{<:STensor{D,R,TS}}, inds::SVector{R}) where {D,R,TS} = unit(STensor{D,R,TS,Float64}, inds)
Defs.unit(F::Type{<:STensor}, inds::Tuple{}) = unit(F, SVector{0,Int}())
Defs.unit(F::Type{<:STensor}, inds::Tuple) = unit(F, SVector(inds))
Defs.unit(F::Type{<:STensor}, inds::Integer...) = unit(F, inds)

Base.:+(x::STensor{D,R,TS}) where {D,R,TS} = STensor{D,R,TS}(+x.elts)
Base.:-(x::STensor{D,R,TS}) where {D,R,TS} = STensor{D,R,TS}(-x.elts)
Base.:+(x::STensor{D,R,TS}, y::STensor{D,R,TS}) where {D,R,TS} = STensor{D,R,TS}(x.elts + y.elts)
Base.:-(x::STensor{D,R,TS}, y::STensor{D,R,TS}) where {D,R,TS} = STensor{D,R,TS}(x.elts - y.elts)
Base.:*(x::STensor{D,R,TS}, a) where {D,R,TS} = STensor{D,R,TS}(x.elts * a)
Base.:/(x::STensor{D,R,TS}, a) where {D,R,TS} = STensor{D,R,TS}(x.elts / a)
Base.:*(a, x::STensor{D,R,TS}) where {D,R,TS} = STensor{D,R,TS}(a * x.elts)
Base.:\(a, x::STensor{D,R,TS}) where {D,R,TS} = STensor{D,R,TS}(a \ x.elts)

################################################################################

# Abstract index notation

#TODO struct Index{D,R}
#TODO     idx::Int
#TODO end

end
