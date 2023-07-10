module DifferentialForms

using Reexport

# The order of these include statements matters
include("Defs.jl")
include("Forms.jl")
include("Multivectors.jl")
include("TensorForms.jl")
include("DoubleForms.jl")

# @reexport using .Defs
using .Defs
export unit, hodge, ⋆, bitsign, sort_perm
@reexport using .Forms
@reexport using .Multivectors
@reexport using .TensorForms
@reexport using .DoubleForms

end
