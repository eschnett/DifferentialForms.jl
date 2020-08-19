using DifferentialForms

using LinearAlgebra
using StaticArrays
using Test

@testset "Create form D=$D R=$R" for D in 0:Dmax, R in 0:D
    T = BigRat

    N = binomial(Val(D), Val(R))
    Form{D,R,T}(SVector{N,T}(rand(T) for n in 1:N))

    fun() = rand(T)
    fun(i) = rand(T)
    fun(i, j) = rand(T)
    fun(i, j, k) = rand(T)
    fun(i, j, k, l) = rand(T)
    fun(i, j, k, l, m) = rand(T)
    Form{D,R,T}(fun::Function)

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
    Form{D,R,T}(arr)
end

@testset "Form D=$D R=$R" for D in 0:Dmax, R in 0:D
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

@testset "Form D=$D R1=$R1 R2=$R2 R3=$R3" for D in 0:Dmax,
R1 in 0:D,
R2 in 0:D,
R3 in 0:D
    # Using === instead of == for comparisons to catch wrong types
    T = Rational{Int64}
    e = one(Form{D,0,T})
    x = rand(Form{D,R1,T})
    y = rand(Form{D,R2,T})
    y2 = rand(Form{D,R2,T})
    z = rand(Form{D,R3,T})
    a = rand(T)
    b = rand(T)

    # Multiplicative structure

    # units
    if D == 2
        e1 = unit(Form{D,1,T}, 1)
        e2 = unit(Form{D,1,T}, 2)
        e12 = unit(Form{D,2,T}, 1)
        @test ⋆e1 === e2
        @test ⋆e2 === -e1
        @test e1 ∧ e1 === 0 * e12
        @test e2 ∧ e2 === 0 * e12
        @test e1 ∧ e2 === e12
    end

    # various duals
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
    if R1 + R2 <= D
        (x ∧ y)::Form{D,R1 + R2}
    end

    if R1 + R2 + R3 <= D
        @test (x ∧ y) ∧ z === x ∧ (y ∧ z)
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
    Rvee = D - ((D - R1) + (D - R2))
    if 0 <= Rvee <= D
        (x ∨ y)::Form{D,Rvee}
        @test ⋆(x ∨ y) === ⋆x ∧ ⋆y
    end

    # dot product: x ⋅ y = x ∨ ⋆y   (right contraction)
    Rdot = D - ((D - R1) + R2)
    if 0 <= Rdot <= D
        (x ⋅ y)::Form{D,Rdot}
        @test x ⋅ y === x ∨ ⋆y
    end

    abs2(x)::T
    @test abs2(x) >= 0
    @test abs2(a * x) == abs2(a) * abs2(x)
    # @test abs2(y + y2) <= abs2(y) + abs2(y2)
    abs(x)::typeof(sqrt(one(T)))
    @test abs(x) ≈ sqrt(abs2(x))
    @test abs(a * x) ≈ abs(a) * abs(x)
    @test abs(y + y2) <= abs(y) + abs(y2) || abs(y + y2) ≈ abs(y) + abs(y2)

    # cross product: x × y = ⋆(x ∧ y)
    Rcross = D - (R1 + R2)
    if 0 <= Rcross
        (x × y)::Form{D,Rcross}
        @test x × y === ⋆(x ∧ y)
    end
end
