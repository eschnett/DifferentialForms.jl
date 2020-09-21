using Test

using Random

const Dmax = 5

# Ignore a statement
macro DISABLED(expr)
    return quote end
end
# # Don't ignore a statement
# macro DISABLED(expr)
#     expr
# end

# Set reproducible random number seed
Random.seed!(0)

# Random rationals
function Base.rand(rng::AbstractRNG,
                   ::Random.SamplerType{Rational{T}}) where {T}
    return Rational{T}(T(rand(rng, -1000:1000))//1000)
end

const BigRat = Rational{BigInt}

@DISABLED include("test-defs.jl")
@DISABLED include("test-indices.jl")
@DISABLED include("test-examples.jl")
@DISABLED include("test-forms.jl")
include("test-multivectors.jl")
