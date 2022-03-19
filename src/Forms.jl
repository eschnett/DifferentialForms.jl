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
Form{D,R}(x::Form{D,R}) where {D,R} = x
Form{D,R}(elts::SVector{N,T}) where {D,R,N,T} = Form{D,R,T}(elts)

# Constructor with added computed type (which must match)
Form{D,R,T,X}(args...) where {D,R,T,X} = Form{D,R,T}(args...)::Form{D,R,T,X}

const IteratorTypes = Union{Base.Generator,Iterators.Flatten}
function Form{D,R,T}(gen::IteratorTypes) where {D,R,T}
    N = length(Form{D,R})
    return Form{D,R,T}(SVector{N,T}(gen))
end
function Form{D,R}(gen::IteratorTypes) where {D,R}
    @assert IteratorEltype(typeof(gen)) == HasEltype()
    N = length(Form{D,R})
    T = eltype(gen)
    return Form{D,R,T}(SVector{N,T}(gen))
end

function Form{D,R,T}(tup::Tuple) where {D,R,T}
    N = length(Form{D,R})
    return Form{D,R,T}(SVector{N,T}(tup))
end
function Form{D,R}(tup::Tuple) where {D,R}
    N = length(Form{D,R})
    return Form{D,R}(SVector{N}(tup))
end
function Form{D,R}(tup::Tuple{}) where {D,R}
    return throw(TypeError("Cannot create Form from emtpy tuple without specifying type"))
end

# Conversions
function Form{D,R,T}(f::Form{D,R}) where {D,R,T}
    return Form{D,R,T}(SVector{length(Form{D,R}),T}(f.elts))
end

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

function Base.show(io::IO, x::Form{D,R,T}) where {D,R,T}
    print(io, "$T{$D,$R}⟦")
    for n in 1:length(x)
        n > 1 && print(io, ", ")
        inds = lin2lst(Val(D), Val(R), n)
        print(io, "[")
        for ind in inds
            print(io, ind)
        end
        print(io, "]:", x[n])
    end
    print(io, "⟧")
    return nothing
end

function Base.show(io::IO, mime::MIME"text/plain", x::Form{D,R,T}) where {D,R,T}
    skiptype = get(io, :typeinfo, Any) <: Form{D,R,T}
    if !skiptype
        print(io, "$T{$D,$R}")
    end
    print(io, "⟦")
    for n in 1:length(x)
        n > 1 && print(io, ", ")
        inds = lin2lst(Val(D), Val(R), n)
        if !get(io, :compact, false)
            print(io, "[")
            for ind in inds
                print(io, ind)
            end
            print(io, "]:")
        end
        show(IOContext(io, :compact => true, :typeinfo => T), mime, x[n])
    end
    print(io, "⟧")
    return nothing
end

################################################################################

# Comparisons

Base.:(==)(x1::Form{D,R}, x2::Form{D,R}) where {D,R} = x1.elts == x2.elts
Base.:(<)(x1::Form{D,R}, x2::Form{D,R}) where {D,R} = x1.elts < x2.elts
Base.isequal(x1::Form, x2::Form) = isequal(x1.elts, x2.elts)
Base.isless(x1::Form, x2::Form) = isless(x1.elts, x2.elts)
Base.hash(x1::Form, h::UInt) = hash(hash(x1.elts, h), UInt(0xc060e76f))
function Base.isapprox(x1::Form{D,R}, x2::Form{D,R}; kw...) where {D,R}
    scale = max(norm(x1), norm(2))
    return isapprox(scale + norm(x1 - x2), scale; kw...)
end

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

export form

# TODO: Use `@generated` only to generate the index lists; access the
# array in a regular funcion
@generated function form(::Val{D}, ::Val{R}, ::Type{T}, arr::AbstractArray{T,R}) where {D,R,T}
    N = binomial(D, R)
    quote
        @assert all(==(D), size(arr))
        elts = SVector{$N,T}($([:(arr[$(lin2lst(Val(D), Val(R), n)...)]) for n in 1:N]...))
        return Form{D,R,T}(elts)
    end
end
function form(::Val{D}, ::Val{R}, ::Type{T}, arr::AbstractArray{U,R}) where {D,R,T,U}
    return Form{D,R,T}(Form{D,R,U}(arr))
end
function form(::Val{D}, ::Val{R}, arr::AbstractArray) where {D,R}
    T = eltype(arr)
    return Form{D,R,T}(arr)
end

function form(::Val{D}, ::Val{R}, ::Type{T}, tup::Tuple) where {D,R,T}
    N = binomial(Val(D), Val(R))
    return Form{D,R,T}(SVector{N,T}(tup))
end
function form(::Val{D}, ::Val{R}, tup::Tuple{}) where {D,R}
    return error("Cannot create Form from emtpy tuple without specifying type")
end
function form(::Val{D}, ::Val{R}, tup::Tuple) where {D,R}
    N = binomial(Val(D), Val(R))
    return Form{D,R}(SVector{N}(tup))
end

function form(::Val{D}, ::Val{R}, ::Type{T}, gen::IteratorTypes) where {D,R,T}
    N = binomial(Val(D), Val(R))
    elts = SVector{N,T}(gen)
    return Form{D,R,T}(elts)
end

# TODO: Use `@generated` only to generate the index lists; call the
# function in a regular funcion
@generated function form(::Val{D}, ::Val{R}, ::Type{T}, fun::Function) where {D,R,T}
    N = binomial(D, R)
    quote
        elts = SVector{$N,T}($([:(fun($(lin2lst(Val(D), Val(R), n)...))::T) for n in 1:N]...))
        return Form{D,R,T}(elts)
    end
end

export MakeForm
struct MakeForm{D,R,T} end
MakeForm{D,R,T}(elts) where {D,R,T} = form(Val(D), Val(R), T, elts)
MakeForm{D,R}(elts) where {D,R} = form(Val(D), Val(R), elts)

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
function uint2bits(::Val{D}, ubits::Integer) where {D}
    return uint2bits(Val(D), Unsigned(ubits))
end

"""
all bit indices for dimension `D` with rank `R`
"""
function dim_bitindices_slow(D::Int, R::Int)
    @assert 0 <= R <= D
    bitindices = Vector{SVector{D,Bool}}()
    for ibits in CartesianIndex(ntuple(d -> 0, D)):CartesianIndex(ntuple(d -> 1, D))
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
Base.size(x::Form, r) = size(typeof(x), r)

Base.getindex(x::Form, ind::Integer) = x.elts[ind]
function Base.getindex(x::Form{D,R}, inds::SVector{R}) where {D,R}
    return x.elts[lst2lin(Val(D), Val(R), inds)]
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
Base.reduce(op, x::Form) = reduce(op, x.elts)
Base.mapreduce(f, op, x::Form{D,R}, ys::Form{D,R}...) where {D,R} = mapreduce(op, f, x.elts, map(y -> y.elts, ys)...)

################################################################################

# Forms form a vector space

function Base.rand(::Type{<:Form{D,R,T}}) where {D,R,T}
    N = binomial(Val(D), Val(R))
    return Form{D,R}(rand(SVector{N,T}))
end
function Base.zero(::Type{<:Form{D,R,T}}) where {D,R,T}
    N = binomial(Val(D), Val(R))
    return Form{D,R}(zero(SVector{N,T}))
end
Base.zero(::Type{<:Form{D,R}}) where {D,R} = zero(Form{D,R,Float64})
Base.zero(x::Form) = zero(typeof(x))
Base.iszero(x::Form) = all(iszero, x.elts)

# Deprecated
@inline Base.zeros(::Type{<:Form{D,R,T}}) where {D,R,T} = zero(Form{D,R,T})
@inline Base.zeros(::Type{<:Form{D,R}}) where {D,R} = zeros(Form{D,R,Float64})
@inline Base.zeros(x::Form) = zeros(typeof(x))

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
Base.conj(x::Form{D,R}) where {D,R} = Form{D,R}(conj(x.elts))
Base.:+(x::Form{D,R}, y::Form{D,R}) where {D,R} = Form{D,R}(x.elts + y.elts)
Base.:-(x::Form{D,R}, y::Form{D,R}) where {D,R} = Form{D,R}(x.elts - y.elts)
Base.:*(x::Form{D,R}, a) where {D,R} = Form{D,R}(x.elts * a)
Base.:/(x::Form{D,R}, a) where {D,R} = Form{D,R}(x.elts / a)
Base.div(x::Form{D,R}, a) where {D,R} = Form{D,R}(map(b -> div(b, a), x.elts))
Base.mod(x::Form{D,R}, a) where {D,R} = Form{D,R}(map(b -> mod(b, a), x.elts))
Base.:*(a, x::Form{D,R}) where {D,R} = Form{D,R}(a * x.elts)
Base.:\(a, x::Form{D,R}) where {D,R} = Form{D,R}(a \ x.elts)

################################################################################

# Forms form an algebra

@inline Base.one(::Type{<:Form{D,R,T}}) where {D,R,T} = one(Form{D,0,T})
@inline Base.one(::Type{<:Form{D,0,T}}) where {D,T} = Form{D,0,T}((one(T),))
@inline Base.one(::Type{<:Form{D,R}}) where {D,R} = one(Form{D,0})
@inline Base.one(::Type{<:Form{D,0}}) where {D} = one(Form{D,0,Float64})
@inline Base.one(::Type{<:Form{D}}) where {D} = one(Form{D,0})
@inline Base.one(x::Form) = one(typeof(x))
@inline Base.isone(x::Form{D,0}) where {D} = isone(x[])

"""
    reverse(x)
    ~x

Reverse
"""
reverse
@inline Base.reverse(x::Form{D,R}) where {D,R} = bitsign((R - 1) * R ÷ 2) * x
@inline Base.:~(x::Form) = reverse(x)
@inline Base.inv(::typeof(~)) = ~

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
hodge(x::Number) = x
@generated function hodge_algorithm(::Val{D}, ::Val{R1}) where {D,R1}
    @assert 0 <= R1 <= D
    R = D - R1
    @assert 0 <= R <= D
    N1 = binomial(D, R1)
    N = binomial(D, R)
    elts = Any[nothing for n in 1:N]
    for n1 in 1:N1
        bits1 = lin2bit(Val(D), Val(R1), n1)
        bitsr = .~bits1
        _, parity = sort_perm(SVector{R1 + R,Int}(bit2lst(Val(D), Val(R1), bits1)..., bit2lst(Val(D), Val(R), bitsr)...))
        s = isodd(parity)
        ind = bit2lin(Val(D), Val(R), bitsr)
        elts[ind] = (s, n1)
    end
    @assert !any(==(nothing), elts)
    return SVector{N,Tuple{Bool,Int}}(elts)
end
@inline @inbounds function eval_hodge_term(term::Tuple{Bool,Int}, x1)
    s, i = term
    return bitsign(s) * ⋆x1[i]
end
function hodge(x1::Form{D,R1}) where {D,R1}
    @assert 0 <= R1 <= D
    R = D - R1
    @assert 0 <= R <= D
    algorithm = hodge_algorithm(Val(D), Val(R1))::SVector
    return Form{D,R}(map(term -> eval_hodge_term(term, x1), algorithm))
end
export hodge, ⋆
const ⋆ = hodge

"""
    invhodge(x)
    inv(⋆)x   (typed: \\star<tab>)

Inverse of Hodge dual: `inv(⋆)⋆x = x`
"""
invhodge
invhodge(x::Number) = x
@generated function invhodge_algorithm(::Val{D}, ::Val{R1}) where {D,R1}
    @assert 0 <= R1 <= D
    R = D - R1
    @assert 0 <= R <= D
    N1 = binomial(D, R1)
    N = binomial(D, R)
    elts = Any[nothing for n in 1:N]
    for n1 in 1:N1
        bits1 = lin2bit(Val(D), Val(R1), n1)
        bitsr = .~bits1
        # The "opposite" parity as `hodge`
        _, parity = sort_perm(SVector{R + R1,Int}(bit2lst(Val(D), Val(R), bitsr)..., bit2lst(Val(D), Val(R1), bits1)...))
        s = isodd(parity)
        ind = bit2lin(Val(D), Val(R), bitsr)
        elts[ind] = (s, n1)
    end
    @assert !any(==(nothing), elts)
    return SVector{N,Tuple{Bool,Int}}(elts)
end
@inline function eval_invhodge_term(term::Tuple{Bool,Int}, x1)
    s, i = term
    @inbounds bitsign(s) * inv(⋆)(x1[i])
end
function invhodge(x1::Form{D,R1}) where {D,R1}
    @assert 0 <= R1 <= D
    R = D - R1
    @assert 0 <= R <= D
    algorithm = invhodge_algorithm(Val(D), Val(R1))::SVector
    return Form{D,R}(map(term -> eval_invhodge_term(term, x1), algorithm))
end
export invhodge
@inline Base.inv(::typeof(hodge)) = invhodge

@inline Base.conj(x::Form{D,R}) where {D,R} = Form{D,R}(conj.(x.elts))

"""
    wedge(x, y)
    x ∧ y   (typed: \\wedge<tab>)

Outer producxt
"""
wedge
wedge(x::Number, ys::Number...) = *(x, ys...)
@generated function wedge_algorithm(::Val{D}, ::Val{R1}, ::Val{R2}) where {D,R1,R2}
    @assert 0 <= R1 <= D
    @assert 0 <= R2 <= D
    R = R1 + R2
    @assert 0 <= R <= D
    N1 = binomial(D, R1)
    N2 = binomial(D, R2)
    N = binomial(D, R)
    elts = [Any[] for n in 1:N]
    for n1 in 1:N1, n2 in 1:N2
        bits1 = lin2bit(Val(D), Val(R1), n1)
        bits2 = lin2bit(Val(D), Val(R2), n2)
        if !any(bits1 .& bits2)
            bitsr = bits1 .| bits2
            _, parity = sort_perm(SVector{R1 + R2,Int}(bit2lst(Val(D), Val(R1), bits1)..., bit2lst(Val(D), Val(R2), bits2)...))
            s = isodd(parity)
            ind = bit2lin(Val(D), Val(R), bitsr)
            push!(elts[ind], (s, n1, n2))
        end
    end
    M = length(elts[1])
    @assert all(elt -> length(elt) == M, elts)
    elts = Tuple.(elts)::Vector{NTuple{M,Tuple{Bool,Int,Int}}}
    return SVector{N,NTuple{M,Tuple{Bool,Int,Int}}}(elts)
end
@inline @inbounds function eval_wedge_term(term::NTuple{M,Tuple{Bool,Int,Int}}, x1, x2) where {M}
    U = typeof(one(eltype(x1)) * one(eltype(x2)))
    M == 0 && return zero(U)
    s, i, j = term[1]
    r = bitsign(s) * (x1[i] ∧ x2[j])
    for m in 2:M
        s, i, j = term[m]
        r += bitsign(s) * (x1[i] ∧ x2[j])
    end
    return r
end
function wedge(x1::Form{D,R1}, x2::Form{D,R2}) where {D,R1,R2}
    @assert 0 <= R1 <= D
    @assert 0 <= R2 <= D
    R = R1 + R2
    @assert 0 <= R <= D
    algorithm = wedge_algorithm(Val(D), Val(R1), Val(R2))::SVector
    return Form{D,R}(map(term -> eval_wedge_term(term, x1, x2), algorithm))
end
@inline wedge(x::Form) = x
@inline wedge(x1::Form, x2::Form, x3s::Form...) = wedge(wedge(x1, x2), x3s...)
@inline wedge(xs::SVector{0,<:Form{D,R,T}}) where {D,R,T} = one(Form{D,0,T})
@inline wedge(xs::SVector{1,<:Form}) = xs[1]
@inline wedge(xs::SVector{N,<:Form}) where {N} = ∧(pop(xs)) ∧ last(xs)
export wedge, ∧
const ∧ = wedge

"""
    vee(x, y)
    x ∨ y   (typed: \\vee<tab>)

Regressive product: `⋆(x ∨ y) = ⋆x ∧ ⋆y`
(Inspired by Grassmann.jl)
"""
vee
vee(x::Number, ys::Number...) = *(x, ys...)
@inline vee(x1::Form, x2::Form) = inv(⋆)(⋆x1 ∧ ⋆x2)
@inline vee(x::Form) = x
@inline vee(x1::Form, x2::Form, x3s::Form...) = vee(vee(x1, x2), x3s...)
@inline vee(xs::SVector{0,<:Form{D,R,T}}) where {D,R,T} = ⋆one(Form{D,0,T})
@inline vee(xs::SVector{1,<:Form}) = xs[1]
@inline vee(xs::SVector{N,<:Form}) where {N} = ∨(pop(xs)) ∨ last(xs)
export vee, ∨
const ∨ = vee

"""
    dot(x, y)
    x ⋅ y   (typed: \\cdot<tab>)

Dot product: `x ⋅ y = x ∨ ⋆y`
(Inspired by Grassmann.jl)
"""
@inline LinearAlgebra.dot(x1::Form, x2::Form) = x1 ∨ ⋆x2
export dot, ⋅

# Base.abs2(x::Form) = (x ⋅ x)[]
# Base.abs(x::Form) = sqrt(abs2(x))
@inline norm2(x::Missing) = missing
@inline norm2(x::Number) = abs2(x)
@inline norm2(x::AbstractArray) = sum(norm2.(x))
# @inline norm2(x::Form) = (x ⋅ x)[]
@inline norm2(x::Form) = norm2(x.elts)
export norm2
@inline LinearAlgebra.norm(x::Form) = sqrt(norm2(x))
export norm

"""
    cross(x, y)
    x × y   (typed: \\times<tab>)

Dot product: `x × y = ⋆(x ∧ y)`
(Inspired by Grassmann.jl)
"""
@inline LinearAlgebra.cross(x1::Form, x2::Form) = ⋆(x1 ∧ x2)
export cross, ×

################################################################################

"""
    tensorsum(x, y)
    x ⊕ y   (typed: \\oplus<tab>)

Tensor sum
"""
tensorsum
@generated function tensorsum_algorithm(::Val{D1}, ::Val{D2}, ::Val{R}) where {D1,D2,R}
    @assert 0 < R <= D1
    @assert 0 < R <= D2
    D = D1 + D2
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
            elts[n] = (Val(1), n1)
        elseif n2 > 0
            @assert n1 == 0
            elts[n] = (Val(2), n2)
        else
            elts[n] = nothing
        end
    end
    return Tuple(elts)
end
@inline function eval_tensorsum_term(term::Nothing, x1, x2)
    U = typeof(one(eltype(x1)) * one(eltype(x2)))
    return zero(U)
end
@inline function eval_tensorsum_term(term::Tuple{Val{I},Int}, x1, x2) where {I}
    _, i = term
    I == 1 && return (@inbounds x1[i])
    return I == 2 && return (@inbounds x2[i])
end
function tensorsum(x1::Form{D1,R}, x2::Form{D2,R}) where {D1,D2,R}
    @assert 0 < R <= D1
    @assert 0 < R <= D2
    D = D1 + D2
    algorithm = tensorsum_algorithm(Val(D1), Val(D2), Val(R))
    return Form{D,R}(map(term -> eval_tensorsum_term(term, x1, x2), algorithm))
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
@generated function tensorproduct_algorithm(::Val{D1}, ::Val{R1}, ::Val{D2}, ::Val{R2}) where {D1,R1,D2,R2}
    @assert 0 <= R1 <= D1
    @assert 0 <= R2 <= D2
    D = D1 + D2
    R = R1 + R2
    N = binomial(D, R)
    elts = Any[nothing for n in 1:N]
    for n in 1:N
        bitsr = lin2bit(Val(D), Val(R), n)
        bits1 = SVector{D1,Bool}(bitsr[1:D1])
        bits2 = SVector{D2,Bool}(bitsr[(D1 + 1):end])
        n1 = bit2lin(Val(D1), Val(R1), bits1)
        n2 = bit2lin(Val(D2), Val(R2), bits2)
        if n1 > 0 && n2 > 0
            elts[n] = (n1, n2)
        else
            elts[n] = nothing
        end
    end
    return Tuple(elts)
end
@inline function eval_tensorproduct_term(term::Nothing, x1, x2)
    U = typeof(one(eltype(x1)) * one(eltype(x2)))
    return zero(U)
end
@inline function eval_tensorproduct_term(term::Tuple{Int,Int}, x1, x2) where {I}
    i, j = term
    return (@inbounds x1[i] * x2[j])
end
function tensorproduct(x1::Form{D1,R1}, x2::Form{D2,R2}) where {D1,R1,D2,R2}
    @assert 0 <= R1 <= D1
    @assert 0 <= R2 <= D2
    D = D1 + D2
    R = R1 + R2
    algorithm = tensorproduct_algorithm(Val(D1), Val(R1), Val(D2), Val(R2))
    return Form{D,R}(map(term -> eval_tensorproduct_term(term, x1, x2), algorithm))
end
tensorproduct(x::Form) = x
function tensorproduct(x1::Form, x2::Form, x3s::Form...)
    return tensorproduct(tensorproduct(x1, x2), x3s...)
end
const ⊗ = tensorproduct
export tensorproduct, ⊗

end
