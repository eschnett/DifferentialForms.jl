module DifferentialForms

using Reexport

# The order of these include statements matters
include("Defs.jl")
include("Forms.jl")

# @reexport using .Defs
using .Defs
export unit, hodge, ⋆, bitsign, sort_perm
@reexport using .Forms

end
