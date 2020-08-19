# Differential Forms

Implement [differential
forms](https://en.wikipedia.org/wiki/Differential_form) in Julia.

* [GitHub](https://github.com/eschnett/DifferentialForms.jl): Source code repository
* [![GitHub CI](https://github.com/eschnett/DifferentialForms.jl/workflows/CI/badge.svg)](https://github.com/eschnett/DifferentialForms.jl/actions)

## Overview

Differential forms are an often very convenient alternative to using
[tensor algebra](https://en.wikipedia.org/wiki/Tensor_algebra) for
multi-dimensional geometric calculations. The fundamental quantity is
an **`R`-form** (a form with rank `R`). In `D` dimensions, `0 ≤ R ≤
D`. `R`-forms are isomorphic to totally antisymmetric rank-`R`
tensors.

In addition to the usual vector operations (add, subtract, scale, dot
product), forms also offer an [exterior
product](https://en.wikipedia.org/wiki/Exterior_algebra#Inner_product)
(or wedge product, written `x ∧ y`) that is equivalent to an
antisymmetrized tensor product, as well as a [hodge
dual](https://en.wikipedia.org/wiki/Hodge_star_operator) (written
`⋆x`, a prefix star operator). Calculating these two operations
efficiently for arbitrary dimensions and ranks is not trivial, and is
the main contribution of this package.

## Examples

We use `D=3` dimensions.

Create some forms:
```Julia
julia> using DifferentialForms

julia> # Create a 0-form (a scalar):
       e = one(Form{3})
Float64⟦1.0⟧{3,0}

julia> e[] == 1
true

julia> collect(e) == [1]
true

julia> # Create a 1-form (a vector) from a tuple:
       v = Form{3,1}((1, 2, 3))
Int64⟦1,2,3⟧{3,1}

julia> v[1] == 1
true

julia> v[2] == 2
true

julia> v[3] == 3
true

julia> collect(v) == [1, 2, 3]
true

julia> # Create a 2-form (an axial vector) from a tuple:
       a = Form{3,2}((1, 2, 3))
Int64⟦1,2,3⟧{3,2}

julia> a[1, 2] == 1
true

julia> a[1, 3] == 2
true

julia> a[2, 3] == 3
true

julia> collect(a) == [1, 2, 3]
true

julia> # Create a 3-form (a pseudoscalar):
       p = Form{3,3}((1,))
Int64⟦1⟧{3,3}

julia> p[1, 2, 3] == 1
true

julia> p[end] == 1
true

julia> collect(p) == [1]
true

julia> # Add, subtract, scale
       a2 = 2*a
Int64⟦2,4,6⟧{3,2}

julia> a3 = a + a2
Int64⟦3,6,9⟧{3,2}

julia> a3 == 3*a
true

julia> # Exterior product (type \wedge<tab>)
       # q = wedge(v, a)
       q = v ∧ a
Float64⟦2.0⟧{3,3}

julia> q == Form{3,3}((2,))
true

julia> # Dot product (type \cdot<tab>)
       # b = dot(v, v)
       b = v ⋅ v
Float64⟦14.0⟧{3,0}

julia> b == 14 * one(Form{3})
true

julia> c = a ⋅ a
Float64⟦14.0⟧{3,0}

julia> c == 14 * one(Form{3})
true

julia> # Cross product (type \times<tab>)
       # w = cross(a, a)
       w = v × v
Float64⟦0.0,-0.0,0.0⟧{3,1}

julia> w == zero(Form{3,1})
true
```

## Related work

This package draws inspiration from
[Grassmann.jl](https://github.com/chakravala/Grassmann.jl), which
includes similar functionality.

[DiscreteDifferentialGeometry.jl](https://github.com/digitaldomain/DiscreteDifferentialGeometry.jl)
also provides similar functionality, but only for two-dimensional
meshes.
