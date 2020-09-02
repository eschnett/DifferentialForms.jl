using DifferentialForms

using LinearAlgebra
using StaticArrays
using Test

@testset "Create forms D=$D R=$R" for D in 0:Dmax, R in 0:D
    T = BigRat

    N = binomial(Val(D), Val(R))
    x = Form{D,R,T}(SVector{N,T}(rand(T) for n in 1:N))
    X = typeof(x)
    x′ = X(SVector{N,T}(rand(T) for n in 1:N))
    if R == 0
        x″ = fscalar(D, rand(T))
        x″::X
    end
    if D > 0 && R == 1
        x″ = fvector(rand(T, D)...)
        x″::X
    end
    if R == D
        x″ = fpseudoscalar(D, rand(T))
        x″::X
    end

    @test Form{D,R,Float64}(x) == Form{D,R,Float64}(x.elts)

    @test Form{D,R,Float64}(collect(x)) == Form{D,R,Float64}(x)
end

@testset "Create forms from arrays D=$D R=$R" for D in 0:Dmax, R in 0:D
    T = BigRat

    fun() = rand(T)
    fun(i) = rand(T)
    fun(i, j) = rand(T)
    fun(i, j, k) = rand(T)
    fun(i, j, k, l) = rand(T)
    fun(i, j, k, l, m) = rand(T)
    y = MakeForm{D,R,T}(fun::Function)

    arr = Array{T}(undef, ntuple(r -> D, R))
    for ind in
        CartesianIndex(ntuple(r -> 1, R)):CartesianIndex(ntuple(r -> D, R))
        lst = SVector{R,Int}(Tuple(ind))
        issorted = true
        for r in 1:(R - 1)
            issorted &= lst[r] < lst[r + 1]
        end
        if issorted
            arr[lst...] = rand(T)
        end
    end
    z = MakeForm{D,R,T}(arr)

    subarr = R == 0 ? (@view arr[]) :
             R == 1 ? (@view arr[:]) :
             R == 2 ? (@view arr[:, :]) :
             R == 3 ? (@view arr[:, :, :]) :
             R == 4 ? (@view arr[:, :, :, :]) :
             R == 5 ? (@view arr[:, :, :, :, :]) : nothing
    subarr::SubArray
    @assert ndims(arr) == R
    @assert ndims(subarr) == R
    @assert all(size(arr, r) == D for r in 1:R)
    @assert all(size(subarr, r) == D for r in 1:R)
    @assert all(==(D), size(arr))
    @assert all(==(D), size(subarr))
    w = MakeForm{D,R,T}(subarr)
    @test w == z
end

@testset "Vector space operations on forms D=$D R=$R" for D in 0:Dmax, R in 0:D
    # Using === instead of == for comparisons to catch wrong types
    T = Rational{Int64}
    n = zero(Form{D,R,T})
    x = rand(Form{D,R,T})
    y = rand(Form{D,R,T})
    z = rand(Form{D,R,T})
    a = rand(T)
    b = rand(T)

    # Vector space

    @test +x === x
    @test (x + y) + z === x + (y + z)

    @test n + x === x
    @test x + n === x

    @test x + y === y + x

    @test x + (-x) === n
    @test (-x) + x === n
    @test x - y === x + (-y)

    @test (a * b) * x === a * (b * x)
    @test x * (a * b) === (x * a) * b

    @test one(T) * a === a
    @test one(T) * x === x
    @test x * one(T) === x

    @test zero(T) * a === zero(T)
    @test zero(T) * x === n
    @test x * zero(T) === n

    if a != zero(T)
        @test a * inv(a) === one(T)

        @test inv(a) * (a * x) === x
        @test (x * a) * inv(a) === x

        @test inv(a) * x === a \ x
        @test x * inv(a) === x / a
    end

    @test (a + b) * x === a * x + b * x
    @test x * (a + b) === x * a + x * b

    @test a * (x + y) === a * x + a * y
    @test (x + y) * a === x * a + y * a

    # Nonlinear transformations
    @test map(+, x) === x
    @test map(+, x, y) === x + y
    @test map(+, x, y, z) === x + y + z
end

@testset "Form algebra D=$D R1=$R1 R2=$R2" for D in 0:Dmax, R1 in 0:D, R2 in 0:D
    # Using === instead of == for comparisons to catch wrong types
    T = Rational{Int64}
    e = one(Form{D,0,T})
    x = rand(Form{D,R1,T})
    y = rand(Form{D,R2,T})
    y2 = rand(Form{D,R2,T})
    a = rand(T)
    b = rand(T)

    # Multiplicative structure

    # units
    if D == 2 && R1 == 1 && R2 == 1
        e = unit(Form{D,0,T})
        e1 = unit(Form{D,1,T}, 1)
        e2 = unit(Form{D,1,T}, 2)
        e12 = unit(Form{D,2,T}, 1, 2)
        @test ~e === e
        @test ~e1 === e1
        @test ~e2 === e2
        @test ~e12 === -e12

        @test ⋆e === e12
        @test ⋆e1 === e2
        @test ⋆e2 === -e1
        @test ⋆e12 === e

        @test e ∧ e === e
        @test e ∧ e1 === e1
        @test e ∧ e2 === e2
        @test e ∧ e12 === e12
        @test e1 ∧ e === e1
        @test e2 ∧ e === e2
        @test e12 ∧ e === e12
        @test e1 ∧ e1 === 0 * e12
        @test e1 ∧ e2 === e12
        @test e2 ∧ e1 === -e12
        @test e2 ∧ e2 === 0 * e12

        @test e ∨ e12 === e
        @test e1 ∨ e1 === 0 * e
        @test e2 ∨ e2 === 0 * e
        @test e1 ∨ e2 === e
        @test e1 ∨ e12 === e1
        @test e2 ∨ e12 === e2
        @test e12 ∨ e12 === e12

        @test e ⋅ e === e
        @test e1 ⋅ e1 === e
        @test e2 ⋅ e2 === e
        @test e1 ⋅ e2 === 0 * e
        @test e12 ⋅ e12 === e

        @test e × e === e12
        @test e × e1 === e2
        @test e × e2 === -e1
        @test e × e12 === e
        @test e1 × e1 === 0 * e
        @test e2 × e2 === 0 * e
        @test e1 × e2 === e
    end

    # various duals
    cycle_basis(x)::typeof(x)
    x′ = x
    for d in 1:D
        x′ = cycle_basis(x′)
    end
    @test x′ === x
    reverse_basis(x)::typeof(x)
    @test reverse_basis(reverse_basis(x)) === x
    @test reverse_basis(cycle_basis(reverse_basis(cycle_basis(x)))) === x

    @test ~~x === x
    @test a * ~x === ~(a * x)

    @test conj(conj(x)) === x
    @test a * conj(x) === conj(a * x)

    @test ⋆⋆x === bitsign(R1 * (D - R1)) * x
    @test inv(⋆)(⋆x) === x
    @test ⋆inv(⋆)(x) === x
    @test ⋆⋆ ⋆ ⋆x === x
    @test a * ⋆x === ⋆(a * x)

    # exterior product: x ∧ y
    @test ∧(x) === x

    if R1 + R2 <= D
        (x ∧ y)::Form{D,R1 + R2}
    end

    for R3 in 0:D
        z = rand(Form{D,R3,T})
        if R1 + R2 + R3 <= D
            @test (x ∧ y) ∧ z === x ∧ (y ∧ z)
            @test x ∧ y ∧ z === (x ∧ y) ∧ z
            @test ∧(x, y, z) === (x ∧ y) ∧ z
        end
    end

    @test e ∧ x === x
    @test x ∧ e === x

    if R1 + R2 <= D
        @test x ∧ zero(y) === zero(x ∧ y)
        @test zero(y) ∧ x === zero(y ∧ x)

        @test a * (x ∧ y) === x ∧ (a * y)
        @test x ∧ (y + y2) === x ∧ y + x ∧ y2

        @test x ∧ y === bitsign(R1 * R2) * (y ∧ x)
    end

    # regressive product: ⋆(x ∨ y) = ⋆x ∧ ⋆y
    @test ∨(x) === x
    Rvee = D - ((D - R1) + (D - R2))
    if 0 <= Rvee <= D
        (x ∨ y)::Form{D,Rvee}
        @test ⋆(x ∨ y) === ⋆x ∧ ⋆y
    end

    for R3 in 0:D
        z = rand(Form{D,R3,T})
        Rvee3 = D - ((D - R1) + (D - R2) + (D - R3))
        if 0 <= Rvee3 <= D
            (x ∨ y ∨ z)::Form{D,Rvee3}
            @test ⋆(x ∨ y ∨ z) === ⋆x ∧ ⋆y ∧ ⋆z
            @test x ∨ y ∨ z === (x ∨ y) ∨ z
            @test ∨(x, y, z) === (x ∨ y) ∨ z
        end
    end

    # dot product: x ⋅ y = x ∨ ⋆y   (right contraction)
    Rdot = D - ((D - R1) + R2)
    if 0 <= Rdot <= D
        (x ⋅ y)::Form{D,Rdot}
        @test x ⋅ y === x ∨ ⋆y
    end

    norm2(x)::T
    @test norm2(x) >= 0
    @test norm2(a * x) == norm2(a) * norm2(x)
    # @test norm2(y + y2) <= norm2(y) + norm2(y2)
    norm(x)::float(T)
    @test norm(x) ≈ sqrt(norm2(x))
    @test norm(a * x) ≈ norm(a) * norm(x)
    @test norm(y + y2) <= norm(y) + norm(y2) ||
          norm(y + y2) ≈ norm(y) + norm(y2)

    # cross product: x × y = ⋆(x ∧ y)
    Rcross = D - (R1 + R2)
    if 0 <= Rcross
        (x × y)::Form{D,Rcross}
        @test x × y === ⋆(x ∧ y)
    end
end

const Dmax4 = min(4, Dmax)
@testset "Tensor sums of forms D1=$D1 D2=$D2 R=$R" for D1 in 0:Dmax4,
D2 in 0:Dmax4,
R in 1:min(D1, D2)

    D = D1 + D2

    T = Rational{Int64}
    x = rand(Form{D1,R,T})
    x2 = rand(Form{D1,R,T})
    y = rand(Form{D2,R,T})
    y2 = rand(Form{D2,R,T})
    a = rand(T)

    # units
    if D1 == 1 && D2 == 1 && R == 1
        u(d, inds...) = unit(Form{d,length(inds),T}, inds...)

        @test u(1, 1) ⊕ u(1, 1) === u(2, 1) + u(2, 2)

        @test u(1, 1) ⊕ u(2, 1) === u(3, 1) + u(3, 2)
        @test u(1, 1) ⊕ u(2, 2) === u(3, 1) + u(3, 3)

        @test u(3, 1) ⊕ u(2, 1) === u(5, 1) + u(5, 4)
        @test u(3, 1) ⊕ u(2, 2) === u(5, 1) + u(5, 5)
        @test u(3, 2) ⊕ u(2, 1) === u(5, 2) + u(5, 4)
        @test u(3, 2) ⊕ u(2, 2) === u(5, 2) + u(5, 5)
        @test u(3, 3) ⊕ u(2, 1) === u(5, 3) + u(5, 4)
        @test u(3, 3) ⊕ u(2, 2) === u(5, 3) + u(5, 5)
    end

    @test ⊕(x) === x
    @test a * ⊕(x) === ⊕(a * x)

    (x ⊕ y)::Form{D,R,T}
    @test a * (x ⊕ y) === (a * x) ⊕ (a * y)
    @test (x ⊕ y) * a === (x * a) ⊕ (y * a)
    @test x ⊕ y === (zero(x) ⊕ y) + (x ⊕ zero(y))
    @test (x + x2) ⊕ (y + y2) === (x ⊕ y) + (x2 ⊕ y2)

    @test reverse_basis(x ⊕ y) ===
          # bitsign(R * R) * 
          (reverse_basis(y) ⊕ reverse_basis(x))

    for D3 in R:Dmax4
        z = rand(Form{D3,R,T})
        @test (x ⊕ y) ⊕ z === x ⊕ (y ⊕ z)
        @test x ⊕ y ⊕ z === x ⊕ (y ⊕ z)
        @test ⊕(x, y, z) === x ⊕ (y ⊕ z)
    end
end

const Dmax3 = min(3, Dmax)
@testset "Tensor products of forms D1=$D1 R1=$R1 D2=$D2 R2=$R2" for D1 in
                                                                    0:Dmax3,
D2 in 0:Dmax3,
R1 in 0:D1,
R2 in 0:D2

    D = D1 + D2
    R = R1 + R2

    T = Rational{Int64}
    e = one(Form{0,0,T})
    x = rand(Form{D1,R1,T})
    x2 = rand(Form{D1,R1,T})
    y = rand(Form{D2,R2,T})
    y2 = rand(Form{D2,R2,T})
    a = rand(T)

    # units
    if D1 == 1 && D2 == 1 && R1 == 1 && R2 == 1
        u(d, inds...) = unit(Form{d,length(inds),T}, inds...)

        @test u(1) ⊗ u(1) === u(2)
        @test u(1, 1) ⊗ u(1) === u(2, 1)
        @test u(1) ⊗ u(1, 1) === u(2, 2)
        @test u(1, 1) ⊗ u(1, 1) === u(2, 1, 2)

        @test u(1) ⊗ u(2) === u(3)
        @test u(1, 1) ⊗ u(2) === u(3, 1)
        @test u(1) ⊗ u(2, 1) === u(3, 2)
        @test u(1, 1) ⊗ u(2, 1) === u(3, 1, 2)
        @test u(1) ⊗ u(2, 2) === u(3, 3)
        @test u(1, 1) ⊗ u(2, 2) === u(3, 1, 3)
        @test u(1) ⊗ u(2, 1, 2) === u(3, 2, 3)
        @test u(1, 1) ⊗ u(2, 1, 2) === u(3, 1, 2, 3)
    end

    @test ⊗(x) === x
    @test a * ⊗(x) === ⊗(a * x)

    (x ⊗ y)::Form{D,R,T}
    @test a * (x ⊗ y) === (a * x) ⊗ y
    @test (x ⊗ y) * a === x ⊗ (y * a)
    @test e ⊗ y === y
    @test x ⊗ e === x
    @test (x + x2) ⊗ y === x ⊗ y + x2 ⊗ y
    @test x ⊗ (y + y2) === x ⊗ y + x ⊗ y2

    @test reverse_basis(x ⊗ y) ===
          bitsign(R1 * R2) * (reverse_basis(y) ⊗ reverse_basis(x))

    for D3 in 0:Dmax3, R3 in 0:D3
        z = rand(Form{D3,R3,T})
        @test (x ⊗ y) ⊗ z === x ⊗ (y ⊗ z)
        @test x ⊗ y ⊗ z === x ⊗ (y ⊗ z)
        @test ⊗(x, y, z) === x ⊗ (y ⊗ z)
    end
end
