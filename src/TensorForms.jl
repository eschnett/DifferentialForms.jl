module TensorForms

using ComputedFieldTypes
using LinearAlgebra
using Random
using StaticArrays

using ..Defs
using ..Forms

"""
The tensor product of two forms, i.e. the composition of two forms,
i.e. a form containing another form
"""
TensorForm

export TensorForm
@computed struct TensorForm{D,R1,R2,T}
    form::fulltype(Form{D,R1,fulltype(Form{D,R2,T})})
end

# Constructor with added computed type (which must match)
TensorForm{D,R1,R2,T,X}(args...) where {D,R1,R2,T,X} = TensorForm{D,R1,R2,T}(args...)::Form{D,R1,R2,T,X}

# Constructor without explicit type
TensorForm{D,R1,R2}(x::TensorForm{D,R1,R2}) where {D,R1,R2} = x
TensorForm{D,R1}(x::TensorForm{D,R1}) where {D,R1} = x
TensorForm{D}(x::TensorForm{D}) where {D} = x
TensorForm(x::TensorForm) = x

TensorForm{D,R1,R2}(form::Form{D,R1,<:Form{D,R2,T}}) where {D,R1,R2,T} = TensorForm{D,R1,R2,T}(form)
TensorForm{D,R1}(form::Form{D,R1,<:Form{D,R2,T}}) where {D,R1,R2,T} = TensorForm{D,R1,R2,T}(form)
TensorForm{D}(form::Form{D,R1,<:Form{D,R2,T}}) where {D,R1,R2,T} = TensorForm{D,R1,R2,T}(form)
TensorForm(form::Form{D,R1,<:Form{D,R2,T}}) where {D,R1,R2,T} = TensorForm{D,R1,R2,T}(form)

# Conversions
function TensorForm{D,R1,R2,T}(x::TensorForm{D,R1,R2}) where {D,R1,R2,T}
    return TensorForm(Form{D,R1,fulltype(Form{D,R2,T})}(x.form))::TensorForm{D,R1,R2,T}
end
function Base.convert(::Type{<:TensorForm{D,R1,R2,T}}, x::TensorForm{D,R1,R2}) where {D,R1,R2,T}
    return TensorForm(Form{D,R1,fulltype(Form{D,R2,T})}(x.form))::TensorForm{D,R1,R2,T}
end

################################################################################

# I/O

function Base.show(io::IO, x::TensorForm{D,R1,R2,T}) where {D,R1,R2,T}
    print(io, "$T{$D,$R1,$R2}⟦")
    for n1 in 1:length(x.form)
        inds1 = Forms.lin2lst(Val(D), Val(R1), n1)
        for n2 in 1:length(x.form[inds1])
            inds2 = Forms.lin2lst(Val(D), Val(R2), n2)
            n1 == n2 == 1 || print(io, ", ")
            print(io, "[")
            for ind1 in inds1
                print(io, ind1)
            end
            print(io, ",")
            for ind2 in inds2
                print(io, ind2)
            end
            print(io, "]:", x.form[n1][n2])
        end
    end
    print(io, "⟧")
    return nothing
end

function Base.show(io::IO, mime::MIME"text/plain", x::TensorForm{D,R1,R2,T}) where {D,R1,R2,T}
    skiptype = get(io, :typeinfo, Any) <: TensorForm{D,R1,R2,T}
    if !skiptype
        print(io, "$T{$D,$R1,$R2}")
    end
    print(io, "⟦")
    for n1 in 1:length(x.form)
        inds1 = Forms.lin2lst(Val(D), Val(R1), n1)
        for n2 in 1:length(x.form[inds1])
            inds2 = Forms.lin2lst(Val(D), Val(R2), n2)
            n1 == n2 == 1 || print(io, ", ")
            if !get(io, :compact, false)
                print(io, "[")
                for ind1 in inds1
                    print(io, ind1)
                end
                print(io, ",")
                for ind2 in inds2
                    print(io, ind2)
                end
                print(io, "]:")
            end
            show(IOContext(io, :compact => true, :typeinfo => T), mime, x.form[n1][n2])
        end
    end
    print(io, "⟧")
    return nothing
end

################################################################################

# Comparisons

Base.:(==)(x::TensorForm{D,R1,R2}, y::TensorForm{D,R1,R2}) where {D,R1,R2} = x.form == y.form

Base.isequal(x::TensorForm, y::TensorForm) = isequal(x.form, y.form)
Base.isless(x::TensorForm, y::TensorForm) = isless(x.form, y.form)

Base.hash(x1::TensorForm, h::UInt) = hash(hash(x1.form, h), UInt(0x398ee043))
function Base.isapprox(x1::TensorForm{D,R1,R2}, x2::TensorForm{D,R1,R2}; kw...) where {D,R1,R2}
    scale = max(norm(x1), norm(2))
    return isapprox(scale + norm(x1 - x2), scale; kw...)
end

################################################################################

# TensorForms are collections

Base.eltype(::Type{<:TensorForm{D,R1,R2,T}}) where {D,R1,R2,T} = T
Base.firstindex(::Type{<:TensorForm}) = 1
Base.firstindex(x::TensorForm) = firstindex(typeof(x))
Base.iterate(x::TensorForm, state...) = iterate(Iterators.flatten(x.form), state...)
Base.ndims(::Type{<:TensorForm{D,R1,R2}}) where {D,R1,R2} = R1 + R2
Base.ndims(x::TensorForm) = ndims(typeof(x))
Base.lastindex(::Type{<:TensorForm{D,R1,R2}}) where {D,R1,R2} = length(TensorForm{D,R1,R2})
Base.lastindex(x::TensorForm) = lastindex(typeof(x))
Base.length(::Type{<:TensorForm{D,R1,R2}}) where {D,R1,R2} = length(Form{D,R1}) * length(Form{D,R2})
Base.length(x::TensorForm) = length(typeof(x))
Base.size(::Type{<:TensorForm{D,R1,R2}}) where {D,R1,R2} = (size(Form{D,R1})..., size(Form{D,R2})...)
Base.size(x::TensorForm) = size(typeof(x))
Base.size(::Type{<:TensorForm{D,R1,R2}}, r) where {D,R1,R2} = size(TensorForm{D,R1,R2})[r]
Base.size(x::TensorForm, r) = size(typeof(x), r)

function Base.getindex(x::TensorForm, ind::Integer)
    length1 = length(x.form)
    length2 = length(x.form[1])
    @assert 1 ≤ ind ≤ length1 * length2
    ind1 = (ind - 1) ÷ length2 + 1
    ind2 = (ind - 1) % length2 + 1
    return x.form[ind1][ind2]
end
function Base.getindex(x::TensorForm{D,R1,R2}, inds::SVector{R12,I}) where {D,R1,R2,R12,I}
    @assert R12 == R1 + R2
    inds1 = SVector{R1,I}(inds[n] for n in 1:R1)
    inds2 = SVector{R2,I}(inds[R1 + n] for n in 1:R2)
    return x.form[inds1][inds2]
end
Base.getindex(x::TensorForm, inds::Tuple{}) = x[SVector{0,Int}()]
Base.getindex(x::TensorForm, inds::Tuple) = x[SVector(inds)]
Base.getindex(x::TensorForm, inds::Integer...) = x[inds]

function Base.setindex(x::TensorForm{D,R1,R2}, val, ind::Integer) where {D,R1,R2}
    length1 = length(x.form)
    length2 = length(x.form[1])
    @assert 1 ≤ ind ≤ length1 * length2
    ind1 = (ind - 1) ÷ length2 + 1
    ind2 = (ind - 1) % length2 + 1
    return TensorForm(setindex(x.form, setindex(x.form[ind1], val, ind2), ind1))::TensorForm{D,R1,R2}
end
function Base.setindex(x::TensorForm{D,R1,R2}, val, inds::SVector{R12,I}) where {D,R1,R2,R12,I}
    @assert R12 == R1 + R2
    inds1 = SVector{R1,I}(inds[n] for n in 1:R1)
    inds2 = SVector{R2,I}(inds[R1 + n] for n in 1:R2)
    return TensorForm(setindex(x.form, setindex(x.form[inds1], val, inds2), inds1))::TensorForm{D,R1,R2}
end
function Base.setindex(x::TensorForm, val, inds::Tuple{})
    return setindex(x, val, SVector{0,Int}())
end
Base.setindex(x::TensorForm, val, inds::Tuple) = setindex(x, val, SVector(inds))
Base.setindex(x::TensorForm, val, inds::Integer...) = setindex(x, val, inds)

function Base.map(f, x::TensorForm{D,R1,R2}, ys::TensorForm{D,R1,R2}...) where {D,R1,R2}
    return TensorForm(map((zs...) -> map(f, zs...), x.form, (y.form for y in ys)...))::TensorForm{D,R1,R2}
end
function Base.mapreduce(f, op, x::TensorForm{D,R1,R2}, ys::TensorForm{D,R1,R2}...; kws...) where {D,R1,R2}
    return mapreduce((zs...) -> mapreduce(f, op, zs...; kws...), op, x.form, (y.form for y in ys)...)
end
function Base.reduce(op, x::TensorForm{D,R1,R2}, ys::TensorForm{D,R1,R2}...; kws...) where {D,R1,R2}
    return mapreduce((zs...) -> reduce(op, zs...; kws...), op, x.form, (y.form for y in ys)...; kws...)
end

################################################################################

# TensorForms form a vector space

function Base.rand(rng::AbstractRNG, ::Random.SamplerType{<:TensorForm{D,R1,R2,T}}) where {D,R1,R2,T}
    return TensorForm(rand(rng, Form{D,R1,fulltype(Form{D,R2,T})}))::TensorForm{D,R1,R2}
end
function Base.zero(::Type{<:TensorForm{D,R1,R2,T}}) where {D,R1,R2,T}
    return TensorForm(zero(Form{D,R1,fulltype(Form{D,R2,T})}))::TensorForm{D,R1,R2}
end
Base.zero(::TensorForm{D,R1,R2,T}) where {D,R1,R2,T} = zero(TensorForm{D,R1,R2,T})
Base.iszero(x::TensorForm) = iszero(x.form)
Base.isreal(x::TensorForm) = isreal(x.form)

function Defs.unit(::Type{<:TensorForm{D,R1,R2,T}}, ind::Integer) where {D,R1,R2,T}
    length1 = length(Form{D,R1})
    length2 = length(Form{D,R2})
    @assert 1 ≤ ind ≤ length1 * length2
    ind1 = (ind - 1) ÷ length2 + 1
    ind2 = (ind - 1) % length2 + 1
    unit2 = unit(Form{D,R2,T}, ind2)
    unit1 = setindex(zero(Form{D,R1,fulltype(Form{D,R2,T})}), unit2, ind1)
    return TensorForm(unit1)::TensorForm{D,R1,R2,T}
end
Defs.unit(::Type{<:TensorForm{D,R1,R2}}, ind::Integer) where {D,R1,R2} = unit(TensorForm{D,R1,R2,Float64}, ind)
function Defs.unit(::Type{<:TensorForm{D,R1,R2,T}}, inds::SVector{R12,I}) where {D,R1,R2,T,R12,I}
    @assert R12 == R1 + R2
    inds1 = SVector{R1,I}(inds[n] for n in 1:R1)
    inds2 = SVector{R2,I}(inds[R1 + n] for n in 1:R2)
    unit2 = unit(Form{D,R2,T}, inds2)
    unit1 = setindex(zero(Form{D,R1,fulltype(Form{D,R2,T})}), unit2, inds1)
    return TensorForm(unit1)::TensorForm{D,R1,R2,T}
end
Defs.unit(::Type{<:TensorForm{D,R1,R2}}, inds::SVector) where {D,R1,R2} = unit(TensorForm{D,R1,R2,Float64}, inds)
Defs.unit(F::Type{<:TensorForm}, inds::Tuple{}) = unit(F, SVector{0,Int}())
Defs.unit(F::Type{<:TensorForm}, inds::Tuple) = unit(F, SVector(inds))
# Defs.unit(F::Type{<:TensorForm}, inds::Integer...) = unit(F, inds)

Base.:+(x::TensorForm) = map(+, x)
Base.:-(x::TensorForm) = map(-, x)
Base.conj(x::TensorForm) = map(conj, x)

Base.:+(x::TensorForm{D,R1,R2}, y::TensorForm{D,R1,R2}) where {D,R1,R2} = map(+, x, y)
Base.:-(x::TensorForm{D,R1,R2}, y::TensorForm{D,R1,R2}) where {D,R1,R2} = map(-, x, y)
Base.:*(a, x::TensorForm) = map(c -> a * c, x)
Base.:*(x::TensorForm, a) = map(c -> c * a, x)
Base.:\(a, x::TensorForm) = map(c -> a \ c, x)
Base.:/(x::TensorForm, a) = map(c -> c / a, x)
Base.div(x::TensorForm, a) = map(c -> div(c, a), x)
Base.mod(x::TensorForm, a) = map(c -> mod(c, a), x)

################################################################################

# TensorForms form an algebra

function Base.one(::Type{<:TensorForm{D,R1,R2,T}}) where {D,R1,R2,T}
    return TensorForm(one(Form{D,R1,fulltype(Form{D,R2,T})}))::TensorForm{D,R1,R2,T}
end
Base.one(::Type{<:TensorForm{D,R1,R2}}) where {D,R1,R2} = one(TensorForm{D,R1,R2,Float64})
Base.one(::Type{<:TensorForm{D}}) where {D} = one(TensorForm{D,0,0})
Base.one(x::TensorForm) = one(typeof(x))
Base.isone(x::TensorForm{D,0,0}) where {D} = isone(x.form)

export swap
function swap(x::TensorForm{D,R1,R2,T}) where {D,R1,R2,T}
    N1 = length(Form{D,R1})
    N2 = length(Form{D,R2})
    return TensorForm(Form{D,R2,fulltype(Form{D,R1,T})}(SVector{N2,fulltype(Form{D,R1,T})}(Form{D,R1,T}(SVector{N1,T}(x.form[n1][n2]
                                                                                                                      for n1 in
                                                                                                                          1:N1))
                                                                                           for n2 in 1:N2)))::TensorForm{D,R2,R1,T}
end
Forms.reverse(x::TensorForm) = TensorForm(~x.form)
Forms.:~(x::TensorForm) = reverse(x)

Forms.hodge(x::TensorForm) = TensorForm(⋆x.form)
Forms.invhodge(x::TensorForm) = TensorForm(inv(⋆)(x.form))

Forms.wedge(x::TensorForm{D}, ys::TensorForm{D}...) where {D} = TensorForm(∧(x.form, (y.form for y in ys)...))::TensorForm{D}
Forms.vee(x::TensorForm{D}, ys::TensorForm{D}...) where {D} = TensorForm(∨(x.form, (y.form for y in ys)...))::TensorForm{D}
LinearAlgebra.dot(x::TensorForm{D}, y::TensorForm{D}) where {D} = TensorForm(x.form ⋅ y.form)::TensorForm{D}
LinearAlgebra.cross(x::TensorForm{D}, y::TensorForm{D}) where {D} = TensorForm(x.form × y.form)::TensorForm{D}

Forms.norm2(x::TensorForm) = norm2(x.form)
LinearAlgebra.norm(x::TensorForm) = norm(x.form)

end
