module DoubleForms

using ComputedFieldTypes
using LinearAlgebra
using Random
using StaticArrays

using ..Defs
using ..Forms
using ..TensorForms: TensorForms, apply1, apply2, swap

"""
The tensor product of two forms, i.e. the composition of two forms,
i.e. a form containing another form
"""
DoubleForm

export DoubleForm
@computed struct DoubleForm{D1,D2,R1,R2,T}
    form::fulltype(Form{D1,R1,fulltype(Form{D2,R2,T})})
end

# Constructor with added computed type (which must match)
DoubleForm{D1,D2,R1,R2,T,X}(args...) where {D1,D2,R1,R2,T,X} = DoubleForm{D1,D2,R1,R2,T}(args...)::DoubleForm{D1,D2,R1,R2,T,X}

# Constructor without explicit type
DoubleForm{D1,D2,R1,R2}(x::DoubleForm{D1,D2,R1,R2}) where {D1,D2,R1,R2} = x
DoubleForm{D1,D2,R1}(x::DoubleForm{D1,D2,R1}) where {D1,D2,R1} = x
DoubleForm{D1,D2}(x::DoubleForm{D1,D2}) where {D1,D2} = x
DoubleForm{D1}(x::DoubleForm{D1}) where {D1} = x
DoubleForm(x::DoubleForm) = x

DoubleForm{D1,D2,R1,R2}(form::Form{D1,R1,<:Form{D2,R2,T}}) where {D1,D2,R1,R2,T} = DoubleForm{D1,D2,R1,R2,T}(form)
DoubleForm{D1,D2,R1}(form::Form{D1,R1,<:Form{D2,R2,T}}) where {D1,D2,R1,R2,T} = DoubleForm{D1,D2,R1,R2,T}(form)
DoubleForm{D1,D2}(form::Form{D1,R1,<:Form{D2,R2,T}}) where {D1,D2,R1,R2,T} = DoubleForm{D1,D2,R1,R2,T}(form)
DoubleForm{D1}(form::Form{D1,R1,<:Form{D2,R2,T}}) where {D1,D2,R1,R2,T} = DoubleForm{D1,D2,R1,R2,T}(form)
DoubleForm(form::Form{D1,R1,<:Form{D2,R2,T}}) where {D1,D2,R1,R2,T} = DoubleForm{D1,D2,R1,R2,T}(form)

# Conversions
function DoubleForm{D1,D2,R1,R2,T}(x::DoubleForm{D1,D2,R1,R2}) where {D1,D2,R1,R2,T}
    return DoubleForm(Form{D1,R1,fulltype(Form{D2,R2,T})}(x.form))::DoubleForm{D1,D2,R1,R2,T}
end
function Base.convert(::Type{<:DoubleForm{D1,D2,R1,R2,T}}, x::DoubleForm{D1,D2,R1,R2}) where {D1,D2,R1,R2,T}
    return DoubleForm(Form{D1,R1,fulltype(Form{D2,R2,T})}(x.form))::DoubleForm{D1,D2,R1,R2,T}
end

################################################################################

# I/O

function Base.show(io::IO, x::DoubleForm{D1,D2,R1,R2,T}) where {D1,D2,R1,R2,T}
    print(io, "$T{$D1,$D2,$R1,$R2}⟦")
    for n1 in 1:length(x.form)
        inds1 = Forms.lin2lst(Val(D1), Val(R1), n1)
        for n2 in 1:length(x.form[inds1])
            inds2 = Forms.lin2lst(Val(D2), Val(R2), n2)
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

function Base.show(io::IO, mime::MIME"text/plain", x::DoubleForm{D1,D2,R1,R2,T}) where {D1,D2,R1,R2,T}
    skiptype = get(io, :typeinfo, Any) <: DoubleForm{D1,D2,R1,R2,T}
    if !skiptype
        print(io, "$T{$D1,$D2,$R1,$R2}")
    end
    print(io, "⟦")
    for n1 in 1:length(x.form)
        inds1 = Forms.lin2lst(Val(D1), Val(R1), n1)
        for n2 in 1:length(x.form[inds1])
            inds2 = Forms.lin2lst(Val(D2), Val(R2), n2)
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

function Base.show(io::IO, mime::MIME"text/latex", x::DoubleForm{D1,D2,R1,R2}) where {D1,D2,R1,R2}
    needsep = false
    for n1 in 1:length(x.form)
        inds1 = Forms.lin2lst(Val(D1), Val(R1), n1)
        for n2 in 1:length(x.form[inds1])
            inds2 = Forms.lin2lst(Val(D2), Val(R2), n2)
            if !iszero(x.form[n1][n2])
                needsep && print(io, " + ")
                needsep = true
                show(io, mime, x.form[n1][n2])
                if !isempty(inds1)
                    print(io, "\\;")
                end
                needsep1 = false
                for ind1 in inds1
                    needsep1 && print(io, " \\wedge ")
                    needsep1 = true
                    print(io, "d", "xyzuvw"[ind1:ind1], "_1")
                end
                if !isempty(inds2)
                    print(io, "\\;")
                end
                needsep2 = false
                for ind2 in inds2
                    needsep2 && print(io, " \\wedge ")
                    needsep2 = true
                    print(io, "d", "xyzuvw"[ind2:ind2], "_2")
                end
            end
        end
    end
    return nothing
end

################################################################################

# Comparisons

Base.:(==)(x::DoubleForm{D1,D2,R1,R2}, y::DoubleForm{D1,D2,R1,R2}) where {D1,D2,R1,R2} = x.form == y.form

Base.isequal(x::DoubleForm, y::DoubleForm) = isequal(x.form, y.form)
Base.isless(x::DoubleForm, y::DoubleForm) = isless(x.form, y.form)

Base.hash(x1::DoubleForm, h::UInt) = hash(hash(x1.form, h), UInt(0x398ee043))
function Base.isapprox(x1::DoubleForm{D1,D2,R1,R2}, x2::DoubleForm{D1,D2,R1,R2}; kw...) where {D1,D2,R1,R2}
    scale = max(norm(x1), norm(2))
    return isapprox(scale + norm(x1 - x2), scale; kw...)
end

################################################################################

# DoubleForms are collections

Base.eltype(::Type{<:DoubleForm{D1,D2,R1,R2,T}}) where {D1,D2,R1,R2,T} = T
Base.firstindex(::Type{<:DoubleForm}) = 1
Base.firstindex(x::DoubleForm) = firstindex(typeof(x))
Base.iterate(x::DoubleForm, state...) = iterate(Iterators.flatten(x.form), state...)
Base.ndims(::Type{<:DoubleForm{D1,D2,R1,R2}}) where {D1,D2,R1,R2} = R1 + R2
Base.ndims(x::DoubleForm) = ndims(typeof(x))
Base.lastindex(::Type{<:DoubleForm{D1,D2,R1,R2}}) where {D1,D2,R1,R2} = length(DoubleForm{D1,D2,R1,R2})
Base.lastindex(x::DoubleForm) = lastindex(typeof(x))
Base.length(::Type{<:DoubleForm{D1,D2,R1,R2}}) where {D1,D2,R1,R2} = length(Form{D1,R1}) * length(Form{D2,R2})
Base.length(x::DoubleForm) = length(typeof(x))
Base.size(::Type{<:DoubleForm{D1,D2,R1,R2}}) where {D1,D2,R1,R2} = (size(Form{D1,R1})..., size(Form{D2,R2})...)
Base.size(x::DoubleForm) = size(typeof(x))
Base.size(::Type{<:DoubleForm{D1,D2,R1,R2}}, r) where {D1,D2,R1,R2} = size(DoubleForm{D1,D2,R1,R2})[r]
Base.size(x::DoubleForm, r) = size(typeof(x), r)

function Base.getindex(x::DoubleForm, ind::Integer)
    length1 = length(x.form)
    length2 = length(x.form[1])
    @assert 1 ≤ ind ≤ length1 * length2
    ind1 = (ind - 1) ÷ length2 + 1
    ind2 = (ind - 1) % length2 + 1
    return x.form[ind1][ind2]
end
function Base.getindex(x::DoubleForm{D1,D2,R1,R2}, inds::SVector{R12,I}) where {D1,D2,R1,R2,R12,I}
    @assert R12 == R1 + R2
    inds1 = SVector{R1,I}(inds[n] for n in 1:R1)
    inds2 = SVector{R2,I}(inds[R1 + n] for n in 1:R2)
    return x.form[inds1][inds2]
end
Base.getindex(x::DoubleForm, inds::Tuple{}) = x[SVector{0,Int}()]
Base.getindex(x::DoubleForm, inds::Tuple) = x[SVector(inds)]
Base.getindex(x::DoubleForm, inds::Integer...) = x[inds]

function Base.setindex(x::DoubleForm{D1,D2,R1,R2}, val, ind::Integer) where {D1,D2,R1,R2}
    length1 = length(x.form)
    length2 = length(x.form[1])
    @assert 1 ≤ ind ≤ length1 * length2
    ind1 = (ind - 1) ÷ length2 + 1
    ind2 = (ind - 1) % length2 + 1
    return DoubleForm(setindex(x.form, setindex(x.form[ind1], val, ind2), ind1))::DoubleForm{D1,D2,R1,R2}
end
function Base.setindex(x::DoubleForm{D1,D2,R1,R2}, val, inds::SVector{R12,I}) where {D1,D2,R1,R2,R12,I}
    @assert R12 == R1 + R2
    inds1 = SVector{R1,I}(inds[n] for n in 1:R1)
    inds2 = SVector{R2,I}(inds[R1 + n] for n in 1:R2)
    return DoubleForm(setindex(x.form, setindex(x.form[inds1], val, inds2), inds1))::DoubleForm{D1,D2,R1,R2}
end
function Base.setindex(x::DoubleForm, val, inds::Tuple{})
    return setindex(x, val, SVector{0,Int}())
end
Base.setindex(x::DoubleForm, val, inds::Tuple) = setindex(x, val, SVector(inds))
Base.setindex(x::DoubleForm, val, inds::Integer...) = setindex(x, val, inds)

function Base.map(f, x::DoubleForm{D1,D2,R1,R2}, ys::DoubleForm{D1,D2,R1,R2}...) where {D1,D2,R1,R2}
    return DoubleForm(map((zs...) -> map(f, zs...), x.form, (y.form for y in ys)...))::DoubleForm{D1,D2,R1,R2}
end
function Base.mapreduce(f, op, x::DoubleForm{D1,D2,R1,R2}, ys::DoubleForm{D1,D2,R1,R2}...; kws...) where {D1,D2,R1,R2}
    return mapreduce((zs...) -> mapreduce(f, op, zs...; kws...), op, x.form, (y.form for y in ys)...)
end
function Base.reduce(op, x::DoubleForm{D1,D2,R1,R2}, ys::DoubleForm{D1,D2,R1,R2}...; kws...) where {D1,D2,R1,R2}
    return mapreduce((zs...) -> reduce(op, zs...; kws...), op, x.form, (y.form for y in ys)...; kws...)
end

function TensorForms.apply1(f, x::DoubleForm{D1,D2,R1,R2}, ys::DoubleForm{D1,D2,R1,R2}...) where {D1,D2,R1,R2}
    return DoubleForm(f(x.form, (y.form for y in ys)...))::DoubleForm{D1,D2,R1,R2}
end
function TensorForms.apply2(f, x::DoubleForm{D1,D2,R1,R2}, ys::DoubleForm{D1,D2,R1,R2}...) where {D1,D2,R1,R2}
    return DoubleForm(map(f, x.form, (y.form for y in ys)...))::DoubleForm{D1,D2,R1,R2}
end

################################################################################

# DoubleForms form a vector space

function Base.rand(rng::AbstractRNG, ::Random.SamplerType{<:DoubleForm{D1,D2,R1,R2,T}}) where {D1,D2,R1,R2,T}
    return DoubleForm(rand(rng, Form{D1,R1,fulltype(Form{D2,R2,T})}))::DoubleForm{D1,D2,R1,R2}
end
function Base.zero(::Type{<:DoubleForm{D1,D2,R1,R2,T}}) where {D1,D2,R1,R2,T}
    return DoubleForm(zero(Form{D1,R1,fulltype(Form{D2,R2,T})}))::DoubleForm{D1,D2,R1,R2}
end
Base.zero(::DoubleForm{D1,D2,R1,R2,T}) where {D1,D2,R1,R2,T} = zero(DoubleForm{D1,D2,R1,R2,T})
Base.iszero(x::DoubleForm) = iszero(x.form)
Base.isreal(x::DoubleForm) = isreal(x.form)

function Defs.unit(::Type{<:DoubleForm{D1,D2,R1,R2,T}}, ind::Integer) where {D1,D2,R1,R2,T}
    length1 = length(Form{D1,R1})
    length2 = length(Form{D2,R2})
    @assert 1 ≤ ind ≤ length1 * length2
    ind1 = (ind - 1) ÷ length2 + 1
    ind2 = (ind - 1) % length2 + 1
    unit2 = unit(Form{D2,R2,T}, ind2)
    unit1 = setindex(zero(Form{D1,R1,fulltype(Form{D2,R2,T})}), unit2, ind1)
    return DoubleForm(unit1)::DoubleForm{D1,D2,R1,R2,T}
end
Defs.unit(::Type{<:DoubleForm{D1,D2,R1,R2}}, ind::Integer) where {D1,D2,R1,R2} = unit(DoubleForm{D1,D2,R1,R2,Float64}, ind)
function Defs.unit(::Type{<:DoubleForm{D1,D2,R1,R2,T}}, inds::SVector{R12,I}) where {D1,D2,R1,R2,T,R12,I}
    @assert R12 == R1 + R2
    inds1 = SVector{R1,I}(inds[n] for n in 1:R1)
    inds2 = SVector{R2,I}(inds[R1 + n] for n in 1:R2)
    unit2 = unit(Form{D2,R2,T}, inds2)
    unit1 = setindex(zero(Form{D1,R1,fulltype(Form{D2,R2,T})}), unit2, inds1)
    return DoubleForm(unit1)::DoubleForm{D1,D2,R1,R2,T}
end
Defs.unit(::Type{<:DoubleForm{D1,D2,R1,R2}}, inds::SVector) where {D1,D2,R1,R2} = unit(DoubleForm{D1,D2,R1,R2,Float64}, inds)
Defs.unit(F::Type{<:DoubleForm}, inds::Tuple{}) = unit(F, SVector{0,Int}())
Defs.unit(F::Type{<:DoubleForm}, inds::Tuple) = unit(F, SVector(inds))
# Defs.unit(F::Type{<:DoubleForm}, inds::Integer...) = unit(F, inds)

Base.:+(x::DoubleForm) = map(+, x)
Base.:-(x::DoubleForm) = map(-, x)
Base.conj(x::DoubleForm) = map(conj, x)

Base.:+(x::DoubleForm{D1,D2,R1,R2}, y::DoubleForm{D1,D2,R1,R2}) where {D1,D2,R1,R2} = map(+, x, y)
Base.:-(x::DoubleForm{D1,D2,R1,R2}, y::DoubleForm{D1,D2,R1,R2}) where {D1,D2,R1,R2} = map(-, x, y)
Base.:*(a, x::DoubleForm) = map(c -> a * c, x)
Base.:*(x::DoubleForm, a) = map(c -> c * a, x)
Base.:\(a, x::DoubleForm) = map(c -> a \ c, x)
Base.:/(x::DoubleForm, a) = map(c -> c / a, x)
Base.div(x::DoubleForm, a) = map(c -> div(c, a), x)
Base.mod(x::DoubleForm, a) = map(c -> mod(c, a), x)

################################################################################

# DoubleForms form an algebra

function Base.one(::Type{<:DoubleForm{D1,D2,R1,R2,T}}) where {D1,D2,R1,R2,T}
    return DoubleForm(one(Form{D1,R1,fulltype(Form{D2,R2,T})}))::DoubleForm{D1,D2,R1,R2,T}
end
Base.one(::Type{<:DoubleForm{D1,D2,R1,R2}}) where {D1,D2,R1,R2} = one(DoubleForm{D1,D2,R1,R2,Float64})
Base.one(::Type{<:DoubleForm{D1,D2}}) where {D1,D2} = one(DoubleForm{D1,D2,0,0})
Base.one(x::DoubleForm) = one(typeof(x))
Base.isone(x::DoubleForm{D1,D2,0,0}) where {D1,D2} = isone(x.form)

function TensorForms.swap(x::DoubleForm{D1,D2,R1,R2,T}) where {D1,D2,R1,R2,T}
    N1 = length(Form{D1,R1})
    N2 = length(Form{D2,R2})
    res = SVector{N2,fulltype(Form{D1,R1,T})}(Form{D1,R1,T}(SVector{N1,T}(x.form[n1][n2] for n1 in 1:N1)) for n2 in 1:N2)
    return DoubleForm(Form{D2,R2,fulltype(Form{D1,R1,T})}(res))::DoubleForm{D2,D1,R2,R1,T}
end
Forms.reverse(x::DoubleForm) = DoubleForm(~x.form)
Forms.:~(x::DoubleForm) = reverse(x)

Forms.hodge(x::DoubleForm) = DoubleForm(⋆x.form)
Forms.invhodge(x::DoubleForm) = DoubleForm(inv(⋆)(x.form))

function Forms.wedge(x::DoubleForm{D1,D2}, ys::DoubleForm{D1,D2}...) where {D1,D2}
    return DoubleForm(∧(x.form, (y.form for y in ys)...))::DoubleForm{D1,D2}
end
function Forms.vee(x::DoubleForm{D1,D2}, ys::DoubleForm{D1,D2}...) where {D1,D2}
    return DoubleForm(∨(x.form, (y.form for y in ys)...))::DoubleForm{D1,D2}
end
LinearAlgebra.dot(x::DoubleForm{D1,D2}, y::DoubleForm{D1,D2}) where {D1,D2} = DoubleForm(x.form ⋅ y.form)::DoubleForm{D1,D2}
LinearAlgebra.cross(x::DoubleForm{D1,D2}, y::DoubleForm{D1,D2}) where {D1,D2} = DoubleForm(x.form × y.form)::DoubleForm{D1,D2}

Forms.norm2(x::DoubleForm) = norm2(x.form)
LinearAlgebra.norm(x::DoubleForm) = norm(x.form)

end
