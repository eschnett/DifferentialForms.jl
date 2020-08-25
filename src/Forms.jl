module Forms

using ComputedFieldTypes
using LinearAlgebra
using StaticArrays

using ..Defs

"""
    Form{D,R,T}

A differential form living in `D` dimensions (`D>=0`) with Rank `R`
(`0<=D<=R`) holding elements of type `T`.
"""
Form

export Form
@computed struct Form{D,R,T}
    elts::SVector{binomial(Val(D), Val(R)),T}
end

# Constructor without explicit type
Form{D,R}(elts::SVector{N,T}) where {D,R,N,T} = Form{D,R,T}(elts)

# Constructor with added computed type (which must match)
Form{D,R,T,X}(args...) where {D,R,T,X} = Form{D,R,T}(args...)::Form{D,R,T,X}

Base.convert(::Type{T}, x::Form{D,0,T}) where {D,T} = x.elts[1]
Base.convert(::Type{SVector}, x::Form{D,1,T}) where {D,T} = x.elts
Base.convert(::Type{SVector{D}}, x::Form{D,1,T}) where {D,T} = x.elts
Base.convert(::Type{SVector{D,T}}, x::Form{D,1,T}) where {D,T} = x.elts

export fdim
fdim(::Type{<:Form{D}}) where {D} = D
fdim(x::Form) = fdim(typeof(x))
export frank
frank(::Type{<:Form{D,R}}) where {D,R} = R
frank(x::Form) = frank(typeof(x))

################################################################################

# I/O

function Base.show(io::IO, x::Form)
    print(io, "$(eltype(x))⟦")
    for n in 1:length(x)
        n > 1 && print(io, ",")
        print(io, x[n])
    end
    print(io, "⟧{$(fdim(x)),$(frank(x))}")
    return nothing
end

################################################################################

# Comparisons

Base.:(==)(x1::Form{D,R}, x2::Form{D,R}) where {D,R} = x1.elts == x2.elts
Base.:(<)(x1::Form{D,R}, x2::Form{D,R}) where {D,R} = x1.elts < x2.elts
function Base.isequal(x1::Form{D,R}, x2::Form{D,R}) where {D,R}
    return isequal(x1.elts, x2.elts)
end
Base.hash(x1::Form, h::UInt) = hash(hash(x1.elts, h), UInt(0xc060e76f))

################################################################################

# Special constructors

export fscalar
"""
    fscalar(Val(D), x)
    fscalar(D, x)

Create a `D`-dimensional scalar (a `0`-form)
"""
fscalar(::Val{D}, x) where {D} = Form{D,0}((x,))
fscalar(D::Integer, x) = fscalar(Val(Int(D)), x)

export fvector
"""
    fvector(x...)

Create a `D`-dimensional vector (a `1`-form) from its `D` elements
"""
fvector(x...) = Form{length(x),1}(SVector(x))

export fpseudoscalar
"""
    fpseudoscalar(Val(D), x)
    fpseudoscalar(D, x)

Create a `D`-dimensional pseudo-scalar (a `D`-form)
"""
fpseudoscalar(::Val{D}, x) where {D} = Form{D,D}((x,))
fpseudoscalar(D::Integer, x) = fpseudoscalar(Val(Int(D)), x)

# Constructors from collections

const IteratorTypes{T,R} = Union{Base.Generator,Iterators.Flatten}
function Form{D,R,T}(gen::IteratorTypes) where {D,R,T}
    N = binomial(Val(D), Val(R))
    elts = SVector{N,T}(gen)
    return Form{D,R,T}(elts)
end

function Form{D,R,T}(tup::Tuple) where {D,R,T}
    N = binomial(Val(D), Val(R))
    return Form{D,R,T}(SVector{N,T}(tup))
end
function Form{D,R}(tup::Tuple{}) where {D,R}
    return @error "Cannot create Form from emtpy tuple without specifying type"
end
function Form{D,R}(tup::Tuple) where {D,R}
    N = binomial(Val(D), Val(R))
    return Form{D,R}(SVector{N}(tup))
end

@generated function Form{D,R,T}(fun::Function) where {D,R,T}
    N = binomial(D, R)
    quote
        elts = SVector{$N,T}($([:(fun($(lin2lst(Val(D), Val(R), n)...)))
                                for n in 1:N]...))
        return Form{D,R,T}(elts)
    end
end

# We cannot use `AbstractArray` here because we need to exclude `SVector`
const ArrayTypes{T,N} = Union{Array{T,N},Adjoint{T,Array{T,N}},
                              Transpose{T,Array{T,N}},SubArray{T,N},
                              Adjoint{T,<:SubArray{T,N}},
                              Transpose{T,<:SubArray{T,N}}}
@generated function Form{D,R,T}(arr::ArrayTypes{T,R}) where {D,R,T}
    N = binomial(D, R)
    quote
        @assert all(==(D), size(arr))
        elts = SVector{$N,T}($([:(arr[$(lin2lst(Val(D), Val(R), n)...)])
                                for n in 1:N]...))
        return Form{D,R,T}(elts)
    end
end
function Form{D,R}(arr::ArrayTypes) where {D,R}
    T = eltype(arr)
    return Form{D,R,T}(arr)
end

################################################################################

"""
We use three different representations for the "tensor indices" for
differential forms. Let us consider a rank `R` form in `D` dimensions.

- The first representation is a list of indices `[i,j,k]`. There are
  `R` indices, and each index is in the range `1:D`. Since the form is
  antisymmetric, we can choose a canonical representations where the
  indices are strictly increasing (`i < j < k`). There can be zero
  indices (for `D = 0`), and there are at most `D` indices (since `R
  <= D`). This representation is easily understood and preferred by
  humans.

- The second representation is a bit mask. Since all indices are
  different and they are ordered, each bit represents whether the
  corresponding index is present or not. For example, for `D = 4`, the
  index list `[1,2,3]` is represented by the bit mask `0111`, and
  `[2,4]` is represented by `1010`. The empty index list `[]` is
  represented by `0000`. This representation makes index manipulations
  (e.g. determining how the elements in a wedge product multiply)
  straightforward.

  This bitwise representation also clarifies that a `D`-dimensional
  form can have `2^D` elements (when summed over all ranks), and that
  a rank-`R` `D`-dimensional form has `R`-choose-`D` components.

- The third representation is a linear index. We number all bit masks
  for a particular rank, and this defines the conversion between a bit
  mask and a linear index. This representation is used to store the
  element of a form in an array.

This is how the elements of a `D = 2` form look in the various
representations:

    rank      linear    bits    indices
    ------------------------------------
    R = 0:    1         0,0     []
    R = 0:    1         0,0     []
    R = 1:    1         0,1     [1]
              2         1,0     [2]
    R = 2:    1         1,1     [1,2]

And here for `D = 3`:

    rank      linear    bits     indices
    ------------------------------------
    R = 0:    1         0,0,0    []
    R = 1:    1         0,0,1    [1]
              2         0,1,0    [2]
              3         1,0,0    [3]
    R = 2:    1         0,1,1    [1,2]
              2         1,0,1    [1,3]
              3         1,1,0    [2,3]
    R = 3:    1         1,1,1    [1,2,3]
"""
indices

# A fast type-level version of `binomial`
@generated function Base.binomial(::Val{D}, ::Val{R}) where {D,R}
    @assert 0 <= R <= D
    return binomial(D, R)
end

function bits2uint(bits::SVector{D,Bool}) where {D}
    ubits = Unsigned(0)
    for d in 1:D
        ubits += Unsigned(bits[d]) << (d - 1)
    end
    return ubits::Unsigned
end

function uint2bits(::Val{D}, ubits::Unsigned) where {D}
    bits = zeros(SVector{D,Bool})
    for d in 1:D
        bits = Base.setindex(bits, (ubits & (Unsigned(1) << (d - 1))) != 0, d)
    end
    return bits::SVector{D,Bool}
end

"""
all bit indices for dimension `D` with rank `R`
"""
function dim_bitindices_slow(D::Int, R::Int)
    @assert 0 <= R <= D
    bitindices = Vector{SVector{D,Bool}}()
    for ibits in
        CartesianIndex(ntuple(d -> 0, D)):CartesianIndex(ntuple(d -> 1, D))
        bits = SVector{D,Bool}(Tuple(ibits))
        if count(bits) == R
            push!(bitindices, bits)
        end
    end
    @assert length(bitindices) == binomial(D, R)
    return bitindices
end
@generated function dim_bitindices(::Val{D}, ::Val{R}) where {D,R}
    return dim_bitindices_slow(D, R)
end

"""
all linear indices for dimension `D` with rank `R`
"""
function dim_linindices_slow(D::Int, R::Int)
    @assert 0 <= R <= D
    bitindices = dim_bitindices(Val(D), Val(R))
    linindices = zeros(Int, 1 << D)
    for (lin, bits) in enumerate(bitindices)
        ubits = bits2uint(bits)
        linindices[ubits + 1] = lin
    end
    @assert count(!=(0), linindices) == binomial(D, R)
    return linindices
end
@generated function dim_linindices(::Val{D}, ::Val{R}) where {D,R}
    return dim_linindices_slow(D, R)
end

"""
convert a linear index to a bit index
"""
function lin2bit(::Val{D}, ::Val{R}, lin::Int) where {D,R}
    return dim_bitindices(Val(D), Val(R))[lin]::SVector{D,Bool}
end

"""
Convert a bit index to a linear index. Return 0 for failure.
"""
function bit2lin(::Val{D}, ::Val{R}, bits::SVector{D,Bool}) where {D,R}
    return dim_linindices(Val(D), Val(R))[bits2uint(bits) + 1]::Int
end

"""
Convert a bit index to an index list
"""
function bit2lst(::Val{D}, ::Val{R}, bits::SVector{D,Bool}) where {D,R}
    list = zero(SVector{R,Int})
    r = 0
    for d in 1:D
        if bits[d]
            r += 1
            list = Base.setindex(list, d, r)
        end
    end
    @assert r == R
    return list::SVector{R,Int}
end

"""
Convert an index list to a bit index
"""
function lst2bit(::Val{D}, ::Val{R}, list::SVector{R,Int}) where {D,R}
    bits = zero(SVector{D,Bool})
    for r in 1:(R - 1)
        @assert list[r] < list[r + 1]
    end
    for r in 1:R
        bits = Base.setindex(bits, true, list[r])
    end
    return bits::SVector{D,Bool}
end

function lin2lst(::Val{D}, ::Val{R}, lin::Int) where {D,R}
    return bit2lst(Val(D), Val(R), lin2bit(Val(D), Val(R), lin))
end

function lst2lin(::Val{D}, ::Val{R}, list::SVector{R,Int}) where {D,R}
    return bit2lin(Val(D), Val(R), lst2bit(Val(D), Val(R), list))
end

################################################################################

# Forms are collections

Base.eltype(::Type{<:Form{D,R,T}}) where {D,R,T} = T
Base.firstindex(::Type{<:Form}) = 1
Base.firstindex(x::Form) = firstindex(typeof(x))
Base.iterate(x::Form, state...) = iterate(x.elts, state...)
Base.ndims(::Type{<:Form{D,R}}) where {D,R} = R
Base.ndims(x::Form) = ndims(typeof(x))
Base.lastindex(::Type{<:Form{D,R}}) where {D,R} = length(Form{D,R})
Base.lastindex(x::Form) = lastindex(typeof(x))
Base.length(::Type{<:Form{D,R}}) where {D,R} = binomial(Val(D), Val(R))
Base.length(x::Form) = length(typeof(x))
Base.size(::Type{<:Form{D,R}}) where {D,R} = ntuple(d -> D, R)
Base.size(x::Form) = size(typeof(x))
Base.size(::Type{<:Form{D}}, r) where {D} = D
Base.size(x::Form, r) = size(typeof(x))

Base.getindex(x::Form, ind::Integer) = x.elts[ind]
function Base.getindex(x::Form{D,R}, inds::SVector{R}) where {D,R}
    return x[lst2lin(Val(D), Val(R), inds)]
end
Base.getindex(x::Form, inds::Tuple{}) = x[SVector{0,Int}()]
Base.getindex(x::Form, inds::Tuple) = x[SVector(inds)]
Base.getindex(x::Form, inds::Integer...) = x[inds]

function Base.setindex(x::Form{D,R}, val, ind::Integer) where {D,R}
    return Form{D,R}(Base.setindex(x.elts, val, ind))
end
function Base.setindex(x::Form{D,R}, val, inds::SVector{R}) where {D,R}
    return Base.setindex(x, val, lst2lin(Val(D), Val(R), inds))
end
function Base.setindex(x::Form, val, inds::Tuple{})
    return Base.setindex(x, val, SVector{0,Int}())
end
Base.setindex(x::Form, val, inds::Tuple) = Base.setindex(x, val, SVector(inds))
Base.setindex(x::Form, val, inds::Integer...) = Base.setindex(x, val, inds)

function Base.map(f, x::Form{D,R}, ys::Form{D,R}...) where {D,R}
    return Form{D,R}(map(f, x.elts, map(y -> y.elts, ys)...))
end
function Base.reduce(f, x::Form{D,R}, ys::Form{D,R}...) where {D,R}
    return reduce(f, x.elts, map(y -> y.elts, ys)...)
end

################################################################################

# Forms form a vector space

function Base.rand(::Type{<:Form{D,R,T}}) where {D,R,T}
    N = binomial(Val(D), Val(R))
    return Form{D,R}(rand(SVector{N,T}))
end
function Base.zeros(::Type{<:Form{D,R,T}}) where {D,R,T}
    N = binomial(Val(D), Val(R))
    return Form{D,R}(zeros(SVector{N,T}))
end
Base.zeros(::Type{<:Form{D,R}}) where {D,R} = zeros(Form{D,R,Float64})
Base.zeros(x::Form) = zeros(typeof(x))

function Defs.unit(::Type{<:Form{D,R,T}}, ind::Integer) where {D,R,T}
    return setindex(zero(Form{D,R,T}), one(T), ind)
end
function Defs.unit(::Type{<:Form{D,R}}, ind::Integer) where {D,R}
    return unit(Form{D,R,Float64}, ind)
end
function Defs.unit(::Type{<:Form{D,R,T}}, inds::SVector{R}) where {D,R,T}
    return setindex(zero(Form{D,R,T}), one(T), inds)
end
function Defs.unit(::Type{<:Form{D,R}}, inds::SVector{R}) where {D,R}
    return unit(Form{D,R,Float64}, inds)
end
Defs.unit(F::Type{<:Form}, inds::Tuple{}) = unit(F, SVector{0,Int}())
Defs.unit(F::Type{<:Form}, inds::Tuple) = unit(F, SVector(inds))
Defs.unit(F::Type{<:Form}, inds::Integer...) = unit(F, inds)

Base.:+(x::Form{D,R}) where {D,R} = Form{D,R}(+x.elts)
Base.:-(x::Form{D,R}) where {D,R} = Form{D,R}(-x.elts)
Base.:+(x::Form{D,R}, y::Form{D,R}) where {D,R} = Form{D,R}(x.elts + y.elts)
Base.:-(x::Form{D,R}, y::Form{D,R}) where {D,R} = Form{D,R}(x.elts - y.elts)
Base.:*(x::Form{D,R}, a) where {D,R} = Form{D,R}(x.elts * a)
Base.:/(x::Form{D,R}, a) where {D,R} = Form{D,R}(x.elts / a)
Base.:*(a, x::Form{D,R}) where {D,R} = Form{D,R}(a * x.elts)
Base.:\(a, x::Form{D,R}) where {D,R} = Form{D,R}(a \ x.elts)

################################################################################

# Forms form an algebra

Base.zero(::Type{<:Form{D,R,T}}) where {D,R,T} = zeros(Form{D,R,T})
Base.zero(::Type{<:Form{D,R}}) where {D,R} = zero(Form{D,R,Float64})
Base.zero(x::Form) = zero(typeof(x))
Base.one(::Type{<:Form{D,0,T}}) where {D,T} = Form{D,0,T}((one(T),))
Base.one(::Type{<:Form{D,0}}) where {D} = one(Form{D,0,Float64})
Base.one(::Type{<:Form{D}}) where {D} = one(Form{D,0})
Base.one(x::Form) = one(typeof(x))

"""
    reverse(x)
    ~x

Reverse
"""
reverse
Base.reverse(x::Form{D,R}) where {D,R} = bitsign((R - 1) * R ÷ 2) * x
Base.:~(x::Form) = reverse(x)
Base.inv(::typeof(~)) = ~

"""
    cycle_basis(x)

Cycle basis: `e_i => e_{i+1}`
"""
cycle_basis
@generated function cycle_basis(x1::Form{D,R}) where {D,R}
    @assert 0 <= R <= D
    U = typeof(one(eltype(x1)))
    N = binomial(D, R)
    elts = Any[nothing for n in 1:N]
    for n1 in 1:length(x1)
        lst1 = lin2lst(Val(D), Val(R), n1)
        lstr0 = mod1.((lst1 .+ 1), D)
        lstr, parity = sort_perm(lstr0)
        s = bitsign(parity)
        op = s > 0 ? :+ : :-
        ind = lst2lin(Val(D), Val(R), lstr)
        elts[ind] = :($op(x1[$n1]))
    end
    @assert !any(==(nothing), elts)
    quote
        Form{D,$R,$U}(SVector{$N,$U}($(elts...)))
    end
end
export cycle_basis

"""
    reverse_basis(x)

Reverse basis: `e_i => e_{D+1-i}`
"""
reverse_basis
@generated function reverse_basis(x1::Form{D,R}) where {D,R}
    @assert 0 <= R <= D
    U = typeof(one(eltype(x1)))
    N = binomial(D, R)
    elts = Any[nothing for n in 1:N]
    for n1 in 1:length(x1)
        lst1 = lin2lst(Val(D), Val(R), n1)
        lstr0 = (D + 1) .- lst1
        lstr, parity = sort_perm(lstr0)
        s = bitsign(parity)
        op = s > 0 ? :+ : :-
        ind = lst2lin(Val(D), Val(R), lstr)
        elts[ind] = :($op(x1[$n1]))
    end
    @assert !any(==(nothing), elts)
    quote
        Form{D,$R,$U}(SVector{$N,$U}($(elts...)))
    end
end
export reverse_basis

"""
    hodge(x)
    ⋆x   (typed: \\star<tab>)

Hodge dual
"""
hodge
@generated function Defs.hodge(x1::Form{D,R1}) where {D,R1}
    @assert 0 <= R1 <= D
    R = D - R1
    @assert 0 <= R <= D
    U = typeof(one(eltype(x1)))
    N = binomial(D, R)
    elts = Any[nothing for n in 1:N]
    for n1 in 1:length(x1)
        bits1 = lin2bit(Val(D), Val(R1), n1)
        bitsr = .~bits1
        _, parity = sort_perm(SVector{R1 + R,Int}(bit2lst(Val(D), Val(R1),
                                                          bits1)...,
                                                  bit2lst(Val(D), Val(R),
                                                          bitsr)...))
        s = bitsign(parity)
        op = s > 0 ? :+ : :-
        ind = bit2lin(Val(D), Val(R), bitsr)
        elts[ind] = :($op(x1[$n1]))
    end
    @assert !any(==(nothing), elts)
    quote
        Form{D,$R,$U}(SVector{$N,$U}($(elts...)))
    end
end

"""
    invhodge(x)
    inv(⋆)x   (typed: \\star<tab>)

Inverse of Hodge dual: `inv(⋆)⋆x = x`
"""
invhodge
@generated function Defs.invhodge(x1::Form{D,R1}) where {D,R1}
    @assert 0 <= R1 <= D
    R = D - R1
    @assert 0 <= R <= D
    U = typeof(one(eltype(x1)))
    N = binomial(D, R)
    elts = Any[:(zero($U)) for n in 1:N]
    for n1 in 1:length(x1)
        bits1 = lin2bit(Val(D), Val(R1), n1)
        bitsr = .~bits1
        # The "opposite" parity as `hodge`
        _, parity = sort_perm(SVector{R + R1,Int}(bit2lst(Val(D), Val(R),
                                                          bitsr)...,
                                                  bit2lst(Val(D), Val(R1),
                                                          bits1)...))
        s = bitsign(parity)
        op = s > 0 ? :+ : :-
        ind = bit2lin(Val(D), Val(R), bitsr)
        elts[ind] = :($op(x1[$n1]))
    end
    quote
        Form{D,$R,$U}(SVector{$N,$U}($(elts...)))
    end
end

Base.conj(x::Form{D,R}) where {D,R} = Form{D,R}(conj.(x.elts))

"""
    wedge(x, y)
    x ∧ y   (typed: \\wedge<tab>)

Outer producxt
"""
wedge

export wedge
@generated function wedge(x1::Form{D,R1}, x2::Form{D,R2}) where {D,R1,R2}
    @assert 0 <= R1 <= D
    @assert 0 <= R2 <= D
    R = R1 + R2
    @assert 0 <= R <= D
    U = typeof(one(eltype(x1)) * one(eltype(x2)) / 1)
    N = binomial(D, R)
    elts = [Any[] for n in 1:N]
    for n1 in 1:length(x1), n2 in 1:length(x2)
        bits1 = lin2bit(Val(D), Val(R1), n1)
        bits2 = lin2bit(Val(D), Val(R2), n2)
        if !any(bits1 .& bits2)
            bitsr = bits1 .| bits2
            _, parity = sort_perm(SVector{R1 + R2,Int}(bit2lst(Val(D), Val(R1),
                                                               bits1)...,
                                                       bit2lst(Val(D), Val(R2),
                                                               bits2)...))
            s = bitsign(parity)
            op = s > 0 ? :+ : :-
            ind = bit2lin(Val(D), Val(R), bitsr)
            push!(elts[ind], :($op(x1[$n1] * x2[$n2])))
        end
    end
    function makesum(elt)
        isempty(elt) && return :(zero($U))
        return :(+($(elt...)))
    end
    quote
        Form{D,$R,$U}(SVector{$N,$U}($(makesum.(elts)...)))
    end
end
wedge(x::Form) = x
wedge(x1::Form, x2::Form, x3s::Form...) = wedge(wedge(x1, x2), x3s...)
export ∧
const ∧ = wedge

"""
    vee(x, y)
    x ∨ y   (typed: \\vee<tab>)

Regressive product: `⋆(x ∨ y) = ⋆x ∧ ⋆y`
(Inspired by Grassmann.jl)
"""
vee
export vee
vee(x1::Form, x2::Form) = inv(⋆)(⋆x1 ∧ ⋆x2)
vee(x::Form) = x
vee(x1::Form, x2::Form, x3s::Form...) = vee(vee(x1, x2), x3s...)
export ∨
const ∨ = vee

"""
    dot(x, y)
    x ⋅ y   (typed: \\cdot<tab>)

Dot product: `x ⋅ y = x ∧ ⋆y`
(Inspired by Grassmann.jl)
"""
LinearAlgebra.dot(x1::Form, x2::Form) = x1 ∨ ⋆x2
export dot, ⋅

# Base.abs2(x::Form) = (x ⋅ x)[]
# Base.abs(x::Form) = sqrt(abs2(x))
norm2(x::Missing) = missing
norm2(x::Number) = abs2(x)
norm2(x::Form) = (x ⋅ x)[]
export norm2
LinearAlgebra.norm(x::Form) = sqrt(norm2(x))
export norm

"""
    cross(x, y)
    x × y   (typed: \\times<tab>)

Dot product: `x × y = ⋆(x ∧ y)`
(Inspired by Grassmann.jl)
"""
LinearAlgebra.cross(x1::Form, x2::Form) = ⋆(x1 ∧ x2)
export cross, ×

################################################################################

"""
    tensorsum(x, y)
    x ⊕ y   (typed: \\oplus<tab>)

Tensor sum
"""
tensorsum
@generated function tensorsum(x1::Form{D1,R}, x2::Form{D2,R}) where {D1,D2,R}
    @assert 0 < R <= D1
    @assert 0 < R <= D2
    D = D1 + D2
    U = typeof(zero(eltype(x1)) + zero(eltype(x2)))
    N = binomial(D, R)
    elts = Any[nothing for n in 1:N]
    for n in 1:N
        bitsr = lin2bit(Val(D), Val(R), n)
        bits1 = SVector{D1,Bool}(bitsr[1:D1])
        bits2 = SVector{D2,Bool}(bitsr[(D1 + 1):end])
        n1 = bit2lin(Val(D1), Val(R), bits1)
        n2 = bit2lin(Val(D2), Val(R), bits2)
        if n1 > 0
            @assert n2 == 0
            elts[n] = :(x1[$n1])
        elseif n2 > 0
            @assert n1 == 0
            elts[n] = :(x2[$n2])
        else
            elts[n] = :(zero($U))
        end
    end
    @assert !any(==(nothing), elts)
    quote
        Form{$D,$R,$U}(SVector{$N,$U}($(elts...)))
    end
end
tensorsum(x::Form) = x
function tensorsum(x1::Form, x2::Form, x3s::Form...)
    return tensorsum(tensorsum(x1, x2), x3s...)
end
const ⊕ = tensorsum
export tensorsum, ⊕

"""
    tensorproduct(x, y)
    x ⊗ y   (typed: \\otimes<tab>)

Tensor product
"""
tensorproduct
@generated function tensorproduct(x1::Form{D1,R1},
                                  x2::Form{D2,R2}) where {D1,R1,D2,R2}
    @assert 0 <= R1 <= D1
    @assert 0 <= R2 <= D2
    D = D1 + D2
    R = R1 + R2
    U = typeof(zero(eltype(x1)) + zero(eltype(x2)))
    N = binomial(D, R)
    elts = Any[nothing for n in 1:N]
    for n in 1:N
        bitsr = lin2bit(Val(D), Val(R), n)
        bits1 = SVector{D1,Bool}(bitsr[1:D1])
        bits2 = SVector{D2,Bool}(bitsr[(D1 + 1):end])
        n1 = bit2lin(Val(D1), Val(R1), bits1)
        n2 = bit2lin(Val(D2), Val(R2), bits2)
        if n1 > 0 && n2 > 0
            elts[n] = :(x1[$n1] * x2[$n2])
        else
            elts[n] = :(zero($U))
        end
    end
    @assert !any(==(nothing), elts)
    quote
        Form{$D,$R,$U}(SVector{$N,$U}($(elts...)))
    end
end
tensorproduct(x::Form) = x
function tensorproduct(x1::Form, x2::Form, x3s::Form...)
    return tensorproduct(tensorproduct(x1, x2), x3s...)
end
const ⊗ = tensorproduct
export tensorproduct, ⊗

end
