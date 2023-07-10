using DifferentialForms

using LinearAlgebra
using StaticArrays
using Test

@testset "Create tensor forms D=$D R1=$R1 R2=$R2" for D in 0:4, R1 in 0:D, R2 in 0:D
    T = BigRat

    x = rand(TensorForm{D,R1,R2,T})

    @test TensorForm{D,R1,R2,Float64}(x) == TensorForm{D,R1,R2,Float64}(x.form)
end

@testset "Comparing tensor forms D=$D R1=$R1 R2=$R2" for D in 0:4, R1 in 0:D, R2 in 0:D
    T = Int

    for n in 1:100
        x = rand(TensorForm{D,R1,R2,T})
        y = rand(TensorForm{D,R1,R2,T})
        isequal(x, y) && continue

        @test x == x
        @test isequal(x, x)
        @test hash(x) == hash(x)
        @test !isless(x, x)

        @test x != y
        @test !isequal(x, y)
        @test hash(x) != hash(y)
        @test isless(x, y) || isless(y, x)
    end
end

@testset "Tensor forms as collections D=$D R1=$R1 R2=$R2" for D in 0:4, R1 in 0:D, R2 in 0:D
    T = Int
    x = rand(TensorForm{D,R1,R2,T})

    N = length(x)

    y = collect(x)
    @test all(y[n] == x[n] for n in 1:N)
    z = [a for a in x]
    @test z == y
    for (i, a) in enumerate(x)
        @test a == x[i]
    end
    w = map(x -> Complex{Int}(x), x)
    @test w == x

    @test sum(x) === sum(y)
    if VERSION ≥ v"1.6"
        @test sum(x; init=0 + 0im) === sum(y; init=0 + 0im)
    end
    @test reduce(+, x) === reduce(+, y)
    @test reduce(+, x; init=0 + 0im) === reduce(+, y; init=0 + 0im)
    @test mapreduce(a -> 2a + 1, +, x) === mapreduce(a -> 2a + 1, +, y)
    @test mapreduce(a -> 2a + 1, +, x; init=0 + 0im) === mapreduce(a -> 2a + 1, +, y; init=0 + 0im)

    scalar(x) = Scalar(x)
    unscalar(x::Scalar) = x[]
    star1(x) = map(unscalar, ⋆map(scalar, x))
    @test apply2(star1, apply1(star1, x)) == ⋆x
    @test apply1(star1, apply2(star1, x)) == apply2(star1, apply1(star1, x))

    N1 = length(x.form)
    N2 = N ÷ N1
    for n in 1:N
        n1 = (n - 1) ÷ N2 + 1
        n2 = (n - 1) % N2 + 1
        y = setindex(x, x[n], n)
        @test y === x

        inds1 = Forms.lin2lst(Val(D), Val(R1), n1)::SVector{R1,Int}
        inds2 = Forms.lin2lst(Val(D), Val(R2), n2)::SVector{R2,Int}
        inds = SVector{R1 + R2,Int}(inds1..., inds2...)
        y = setindex(x, x[inds], inds)
        @test y === x

        tup = Tuple(inds)::NTuple{R1 + R2,Int}
        y = setindex(x, x[tup], tup)
        @test y === x

        y = setindex(x, x[inds...], inds...)
        @test y === x
    end
end

@testset "Vector space operations on forms D=$D R1=$R1 R2=$R2" for D in 0:4, R1 in 0:D, R2 in 0:D
    # Using === instead of == for comparisons to catch wrong types
    T = Rational{Int64}
    n = zero(TensorForm{D,R1,R2,T})
    x = rand(TensorForm{D,R1,R2,T})
    y = rand(TensorForm{D,R1,R2,T})
    z = rand(TensorForm{D,R1,R2,T})
    a = rand(T)
    b = rand(T)

    # Vector space

    @test iszero(n)
    @test iszero(x) === (x === n)

    @test +x === x
    @test (x + y) + z === x + (y + z)

    @test n + x === x
    @test x + n === x

    @test x + y === y + x

    @test x + (-x) === n
    @test (-x) + x === n
    @test x - y === x + (-y)

    @test conj(conj(x)) === x

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

@testset "Tensor form algebra D=$D R1x=$R1x R1y=$R1y R2x=$R2x R2y=$R2y" for D in 0:3, R1x in 0:D, R1y in 0:D, R2x in 0:D, R2y in 0:D
    # Using === instead of == for comparisons to catch wrong types
    T = Rational{Int64}
    e = one(TensorForm{D,0,0,T})
    x = rand(TensorForm{D,R1x,R2x,T})
    y = rand(TensorForm{D,R1y,R2y,T})
    y2 = rand(TensorForm{D,R1y,R2y,T})
    a = rand(T)
    b = rand(T)

    # Multiplicative structure

    @test isone(e)
    @test !isone(2 * e)

    # units
    if D == 2 && R1x == 0 && R1y == 0 && R2x == 0 && R2y == 0
        e00 = unit(TensorForm{D,0,0,T}, ())
        e10 = unit(TensorForm{D,1,0,T}, (1,))
        e20 = unit(TensorForm{D,1,0,T}, (2,))
        e30 = unit(TensorForm{D,2,0,T}, (1, 2))
        e01 = unit(TensorForm{D,0,1,T}, (1,))
        e11 = unit(TensorForm{D,1,1,T}, (1, 1))
        e21 = unit(TensorForm{D,1,1,T}, (2, 1))
        e31 = unit(TensorForm{D,2,1,T}, (1, 2, 1))
        e02 = unit(TensorForm{D,0,1,T}, (2,))
        e12 = unit(TensorForm{D,1,1,T}, (1, 2))
        e22 = unit(TensorForm{D,1,1,T}, (2, 2))
        e32 = unit(TensorForm{D,2,1,T}, (1, 2, 2))
        e03 = unit(TensorForm{D,0,2,T}, (1, 2))
        e13 = unit(TensorForm{D,1,2,T}, (1, 1, 2))
        e23 = unit(TensorForm{D,1,2,T}, (2, 1, 2))
        e33 = unit(TensorForm{D,2,2,T}, (1, 2, 1, 2))

        @test ~e00 === +e00
        @test ~e10 === +e10
        @test ~e20 === +e20
        @test ~e30 === -e30
        @test ~e01 === +e01
        @test ~e11 === +e11
        @test ~e21 === +e21
        @test ~e31 === -e31
        @test ~e02 === +e02
        @test ~e12 === +e12
        @test ~e22 === +e22
        @test ~e32 === -e32
        @test ~e03 === -e03
        @test ~e13 === -e13
        @test ~e23 === -e23
        @test ~e33 === +e33

        @test ⋆e00 === +e33
        @test ⋆e10 === +e23
        @test ⋆e20 === -e13
        @test ⋆e30 === +e03
        @test ⋆e01 === +e32
        @test ⋆e11 === +e22
        @test ⋆e21 === -e12
        @test ⋆e31 === +e02
        @test ⋆e02 === -e31
        @test ⋆e12 === -e21
        @test ⋆e22 === +e11
        @test ⋆e32 === -e01
        @test ⋆e03 === +e30
        @test ⋆e13 === +e20
        @test ⋆e23 === -e10
        @test ⋆e33 === +e00

        @test e00 ∧ e00 === e00
        @test e00 ∧ e10 === e10
        @test e00 ∧ e20 === e20
        @test e00 ∧ e30 === e30
        @test e00 ∧ e01 === e01
        @test e00 ∧ e11 === e11
        @test e00 ∧ e21 === e21
        @test e00 ∧ e31 === e31
        @test e00 ∧ e02 === e02
        @test e00 ∧ e12 === e12
        @test e00 ∧ e22 === e22
        @test e00 ∧ e32 === e32
        @test e00 ∧ e03 === e03
        @test e00 ∧ e13 === e13
        @test e00 ∧ e23 === e23
        @test e00 ∧ e33 === e33

        @test e10 ∧ e00 === e10
        @test e10 ∧ e10 === 0 * e30
        @test e10 ∧ e20 === e30
        @test_throws AssertionError e10 ∧ e30
        @test e10 ∧ e01 === e11
        @test e10 ∧ e11 === 0 * e31
        @test e10 ∧ e21 === e31
        @test_throws AssertionError e10 ∧ e31
        @test e10 ∧ e02 === e12
        @test e10 ∧ e12 === 0 * e31
        @test e10 ∧ e22 === e32
        @test_throws AssertionError e10 ∧ e32
        @test e10 ∧ e03 === e13
        @test e10 ∧ e13 === 0 * e33
        @test e10 ∧ e23 === e33
        @test_throws AssertionError e10 ∧ e33

        @test e20 ∧ e00 === e20
        @test e20 ∧ e10 === -e30
        @test e20 ∧ e20 === 0 * e30
        @test_throws AssertionError e20 ∧ e30
        @test e20 ∧ e01 === e21
        @test e20 ∧ e11 === -e31
        @test e20 ∧ e21 === 0 * e31
        @test_throws AssertionError e20 ∧ e31
        @test e20 ∧ e02 === e22
        @test e20 ∧ e12 === -e32
        @test e20 ∧ e22 === 0 * e32
        @test_throws AssertionError e20 ∧ e32
        @test e20 ∧ e03 === e23
        @test e20 ∧ e13 === -e33
        @test e20 ∧ e23 === 0 * e33
        @test_throws AssertionError e20 ∧ e33

        @test e30 ∧ e00 === e30
        @test e30 ∧ e01 === e31
        @test e30 ∧ e02 === e32
        @test e30 ∧ e03 === e33
        @test_throws AssertionError e30 ∧ e10
        @test_throws AssertionError e30 ∧ e11
        @test_throws AssertionError e30 ∧ e12
        @test_throws AssertionError e30 ∧ e13
        @test_throws AssertionError e30 ∧ e20
        @test_throws AssertionError e30 ∧ e21
        @test_throws AssertionError e30 ∧ e22
        @test_throws AssertionError e30 ∧ e23
        @test_throws AssertionError e30 ∧ e30
        @test_throws AssertionError e30 ∧ e31
        @test_throws AssertionError e30 ∧ e32
        @test_throws AssertionError e30 ∧ e33

        @test e01 ∧ e00 === e01
        @test e01 ∧ e10 === e11
        @test e01 ∧ e20 === e21
        @test e01 ∧ e30 === e31
        @test e01 ∧ e01 === 0 * e03
        @test e01 ∧ e11 === 0 * e13
        @test e01 ∧ e21 === 0 * e23
        @test e01 ∧ e31 === 0 * e33
        @test e01 ∧ e02 === e03
        @test e01 ∧ e12 === e13
        @test e01 ∧ e22 === e23
        @test e01 ∧ e32 === e33
        @test_throws AssertionError e01 ∧ e03
        @test_throws AssertionError e01 ∧ e13
        @test_throws AssertionError e01 ∧ e23
        @test_throws AssertionError e01 ∧ e33

        @test e11 ∧ e00 === e11
        @test e11 ∧ e10 === 0 * e31
        @test e11 ∧ e20 === e31
        @test_throws AssertionError e11 ∧ e30
        @test e11 ∧ e01 === 0 * e13
        @test e11 ∧ e11 === 0 * e33
        @test e11 ∧ e21 === 0 * e33
        @test_throws AssertionError e11 ∧ e31
        @test e11 ∧ e02 === e13
        @test e11 ∧ e12 === 0 * e33
        @test e11 ∧ e22 === e33
        @test_throws AssertionError e11 ∧ e32
        @test_throws AssertionError e11 ∧ e03
        @test_throws AssertionError e11 ∧ e13
        @test_throws AssertionError e11 ∧ e23
        @test_throws AssertionError e11 ∧ e33

        @test e21 ∧ e00 === e21
        @test e21 ∧ e10 === -e31
        @test e21 ∧ e20 === 0 * e31
        @test_throws AssertionError e21 ∧ e30
        @test e21 ∧ e01 === 0 * e13
        @test e21 ∧ e11 === 0 * e33
        @test e21 ∧ e21 === 0 * e33
        @test_throws AssertionError e21 ∧ e31
        @test e21 ∧ e02 === e23
        @test e21 ∧ e12 === -e33
        @test e21 ∧ e22 === 0 * e33
        @test_throws AssertionError e21 ∧ e32
        @test_throws AssertionError e21 ∧ e03
        @test_throws AssertionError e21 ∧ e13
        @test_throws AssertionError e21 ∧ e23
        @test_throws AssertionError e21 ∧ e33

        @test e31 ∧ e00 === e31
        @test_throws AssertionError e31 ∧ e10
        @test_throws AssertionError e31 ∧ e20
        @test_throws AssertionError e31 ∧ e30
        @test e31 ∧ e01 === 0 * e33
        @test_throws AssertionError e31 ∧ e11
        @test_throws AssertionError e31 ∧ e21
        @test_throws AssertionError e31 ∧ e31
        @test e31 ∧ e02 === e33
        @test_throws AssertionError e31 ∧ e12
        @test_throws AssertionError e31 ∧ e22
        @test_throws AssertionError e31 ∧ e32
        @test_throws AssertionError e31 ∧ e03
        @test_throws AssertionError e31 ∧ e13
        @test_throws AssertionError e31 ∧ e23
        @test_throws AssertionError e31 ∧ e33

        @test e02 ∧ e00 === e02
        @test e02 ∧ e10 === e12
        @test e02 ∧ e20 === e22
        @test e02 ∧ e30 === e32
        @test e02 ∧ e01 === -e03
        @test e02 ∧ e11 === -e13
        @test e02 ∧ e21 === -e23
        @test e02 ∧ e31 === -e33
        @test e02 ∧ e02 === 0 * e03
        @test e02 ∧ e12 === 0 * e13
        @test e02 ∧ e22 === 0 * e13
        @test e02 ∧ e32 === 0 * e33
        @test_throws AssertionError e02 ∧ e03
        @test_throws AssertionError e02 ∧ e13
        @test_throws AssertionError e02 ∧ e23
        @test_throws AssertionError e02 ∧ e33

        @test e12 ∧ e00 === e12
        @test e12 ∧ e10 === 0 * e32
        @test e12 ∧ e20 === e32
        @test_throws AssertionError e12 ∧ e30
        @test e12 ∧ e01 === -e13
        @test e12 ∧ e11 === 0 * e33
        @test e12 ∧ e21 === -e33
        @test_throws AssertionError e12 ∧ e31
        @test e12 ∧ e02 === 0 * e13
        @test e12 ∧ e12 === 0 * e33
        @test e12 ∧ e22 === 0 * e33
        @test_throws AssertionError e12 ∧ e32
        @test_throws AssertionError e12 ∧ e03
        @test_throws AssertionError e12 ∧ e13
        @test_throws AssertionError e12 ∧ e23
        @test_throws AssertionError e12 ∧ e33

        @test e22 ∧ e00 === e22
        @test e22 ∧ e10 === -e32
        @test e22 ∧ e20 === 0 * e32
        @test_throws AssertionError e22 ∧ e30
        @test e22 ∧ e01 === -e23
        @test e22 ∧ e11 === e33
        @test e22 ∧ e21 === 0 * e33
        @test_throws AssertionError e22 ∧ e31
        @test e22 ∧ e02 === 0 * e13
        @test e22 ∧ e12 === 0 * e33
        @test e22 ∧ e22 === 0 * e33
        @test_throws AssertionError e22 ∧ e32
        @test_throws AssertionError e22 ∧ e03
        @test_throws AssertionError e22 ∧ e13
        @test_throws AssertionError e22 ∧ e23
        @test_throws AssertionError e22 ∧ e33

        @test e32 ∧ e00 === e32
        @test_throws AssertionError e32 ∧ e10
        @test_throws AssertionError e32 ∧ e20
        @test_throws AssertionError e32 ∧ e30
        @test e32 ∧ e01 === -e33
        @test_throws AssertionError e32 ∧ e11
        @test_throws AssertionError e32 ∧ e21
        @test_throws AssertionError e32 ∧ e31
        @test e32 ∧ e02 === 0 * e33
        @test_throws AssertionError e32 ∧ e12
        @test_throws AssertionError e32 ∧ e22
        @test_throws AssertionError e32 ∧ e32
        @test_throws AssertionError e32 ∧ e03
        @test_throws AssertionError e32 ∧ e13
        @test_throws AssertionError e32 ∧ e23
        @test_throws AssertionError e32 ∧ e33

        @test e03 ∧ e00 === e03
        @test e03 ∧ e10 === e13
        @test e03 ∧ e20 === e23
        @test e03 ∧ e30 === e33
        @test_throws AssertionError e03 ∧ e01
        @test_throws AssertionError e03 ∧ e11
        @test_throws AssertionError e03 ∧ e21
        @test_throws AssertionError e03 ∧ e31
        @test_throws AssertionError e03 ∧ e02
        @test_throws AssertionError e03 ∧ e12
        @test_throws AssertionError e03 ∧ e22
        @test_throws AssertionError e03 ∧ e32
        @test_throws AssertionError e03 ∧ e03
        @test_throws AssertionError e03 ∧ e13
        @test_throws AssertionError e03 ∧ e23
        @test_throws AssertionError e03 ∧ e33

        @test e13 ∧ e00 === e13
        @test e13 ∧ e10 === 0 * e33
        @test e13 ∧ e20 === e33
        @test_throws AssertionError e13 ∧ e30
        @test_throws AssertionError e13 ∧ e01
        @test_throws AssertionError e13 ∧ e11
        @test_throws AssertionError e13 ∧ e21
        @test_throws AssertionError e13 ∧ e31
        @test_throws AssertionError e13 ∧ e02
        @test_throws AssertionError e13 ∧ e12
        @test_throws AssertionError e13 ∧ e22
        @test_throws AssertionError e13 ∧ e32
        @test_throws AssertionError e13 ∧ e03
        @test_throws AssertionError e13 ∧ e13
        @test_throws AssertionError e13 ∧ e23
        @test_throws AssertionError e13 ∧ e33

        @test e23 ∧ e00 === e23
        @test e23 ∧ e10 === -e33
        @test e23 ∧ e20 === 0 * e33
        @test_throws AssertionError e23 ∧ e30
        @test_throws AssertionError e23 ∧ e01
        @test_throws AssertionError e23 ∧ e11
        @test_throws AssertionError e23 ∧ e21
        @test_throws AssertionError e23 ∧ e31
        @test_throws AssertionError e23 ∧ e02
        @test_throws AssertionError e23 ∧ e12
        @test_throws AssertionError e23 ∧ e22
        @test_throws AssertionError e23 ∧ e32
        @test_throws AssertionError e23 ∧ e03
        @test_throws AssertionError e23 ∧ e13
        @test_throws AssertionError e23 ∧ e23
        @test_throws AssertionError e23 ∧ e33

        @test e33 ∧ e00 === e33
        @test_throws AssertionError e33 ∧ e10
        @test_throws AssertionError e33 ∧ e20
        @test_throws AssertionError e33 ∧ e30
        @test_throws AssertionError e33 ∧ e01
        @test_throws AssertionError e33 ∧ e11
        @test_throws AssertionError e33 ∧ e21
        @test_throws AssertionError e33 ∧ e31
        @test_throws AssertionError e33 ∧ e02
        @test_throws AssertionError e33 ∧ e12
        @test_throws AssertionError e33 ∧ e22
        @test_throws AssertionError e33 ∧ e32
        @test_throws AssertionError e33 ∧ e03
        @test_throws AssertionError e33 ∧ e13
        @test_throws AssertionError e33 ∧ e23
        @test_throws AssertionError e33 ∧ e33

        # @test e ∨ e12 === e
        # @test e1 ∨ e1 === 0 * e
        # @test e2 ∨ e2 === 0 * e
        # @test e1 ∨ e2 === e
        # @test e1 ∨ e12 === e1
        # @test e2 ∨ e12 === e2
        # @test e12 ∨ e12 === e12
        # 
        # @test e ⋅ e === e
        # @test e1 ⋅ e1 === e
        # @test e2 ⋅ e2 === e
        # @test e1 ⋅ e2 === 0 * e
        # @test e12 ⋅ e12 === e
        # 
        # @test e × e === e12
        # @test e × e1 === e2
        # @test e × e2 === -e1
        # @test e × e12 === e
        # @test e1 × e1 === 0 * e
        # @test e2 × e2 === 0 * e
        # @test e1 × e2 === e
    end

    # # various duals
    # cycle_basis(x)::typeof(x)
    # x′ = x
    # for d in 1:D
    #     x′ = cycle_basis(x′)
    # end
    # @test x′ === x
    # reverse_basis(x)::typeof(x)
    # @test reverse_basis(reverse_basis(x)) === x
    # @test reverse_basis(cycle_basis(reverse_basis(cycle_basis(x)))) === x

    @test swap(x) isa TensorForm{D,R2x,R1x,T}
    @test swap(swap(x)) === x

    @test ~~x === x
    @test a * ~x === ~(a * x)

    @test conj(conj(x)) === x
    @test a * conj(x) === conj(a * x)

    @test ⋆⋆x === bitsign(R1x * (D - R1x)) * bitsign(R2x * (D - R2x)) * x
    @test inv(⋆)(⋆x) === x
    @test ⋆inv(⋆)(x) === x
    @test ⋆⋆ ⋆ ⋆x === x
    @test a * ⋆x === ⋆(a * x)

    # exterior product: x ∧ y
    @test ∧(x) === x

    if R1x + R1y <= D && R2x + R2y <= D
        (x ∧ y)::TensorForm{D,R1x + R1y,R2x + R2y}
    end

    for R1z in 0:D, R2z in 0:D
        z = rand(TensorForm{D,R1z,R2z,T})
        if R1x + R1y + R1z <= D && R2x + R2y + R2z <= D
            @test (x ∧ y) ∧ z === x ∧ (y ∧ z)
            @test x ∧ y ∧ z === (x ∧ y) ∧ z
            @test ∧(x, y, z) === (x ∧ y) ∧ z
            # @test ∧(SVector{0,typeof(x)}()) === one(x)
            # @test ∧(SVector(x)) === x
            # @test ∧(SVector(x, y)) === x ∧ y
            # @test ∧(SVector(x, y, z)) === (x ∧ y) ∧ z
            # @test ∧(SVector{0,typeof(x)}()) ∧ x === x
            # @test x ∧ ∧(SVector{0,typeof(x)}()) === x
        end
    end

    @test e ∧ x === x
    @test x ∧ e === x

    R1wedge = R1x + R1y
    R2wedge = R2x + R2y
    if 0 <= R1wedge <= D && 0 <= R2wedge <= D
        @test x ∧ zero(y) === zero(x ∧ y)
        @test zero(y) ∧ x === zero(y ∧ x)

        @test a * (x ∧ y) === x ∧ (a * y)
        @test x ∧ (y + y2) === x ∧ y + x ∧ y2

        @test x ∧ y === bitsign(R1x * R1y) * bitsign(R2x * R2y) * (y ∧ x)
    end

    # regressive product: ⋆(x ∨ y) = ⋆x ∧ ⋆y
    @test ∨(x) === x
    R1vee = D - ((D - R1x) + (D - R1y))
    R2vee = D - ((D - R2x) + (D - R2y))
    @test R1vee == R1x + R1y - D
    @test R2vee == R2x + R2y - D
    if 0 <= R1vee <= D && 0 <= R2vee <= D
        (x ∨ y)::TensorForm{D,R1vee,R2vee}
        @test ⋆(x ∨ y) === ⋆x ∧ ⋆y
    end

    for R1z in 0:D, R2z in 0:D
        z = rand(TensorForm{D,R1z,R2z,T})
        R1veez = D - ((D - R1x) + (D - R1y) + (D - R1z))
        R2veez = D - ((D - R2x) + (D - R2y) + (D - R2z))
        if 0 <= R1veez <= D && 0 <= R2veez <= D
            (x ∨ y ∨ z)::TensorForm{D,R1veez,R2veez}
            @test ⋆(x ∨ y ∨ z) === ⋆x ∧ ⋆y ∧ ⋆z
            @test x ∨ y ∨ z === (x ∨ y) ∨ z
            @test ∨(x, y, z) === (x ∨ y) ∨ z
            # @test ∨(SVector{0,typeof(x)}()) === ⋆one(x)
            # @test ∨(SVector(x)) === x
            # @test ∨(SVector(x, y)) === x ∨ y
            # @test ∨(SVector(x, y, z)) === (x ∨ y) ∨ z
            # @test ∨(SVector{0,typeof(x)}()) ∨ x === x
            # @test x ∨ ∨(SVector{0,typeof(x)}()) === x
        end
    end

    # dot product: x ⋅ y = x ∨ ⋆y   (right contraction)
    R1dot = D - ((D - R1x) + R1y)
    R2dot = D - ((D - R2x) + R2y)
    @test R1dot == R1x - R1y
    @test R2dot == R2x - R2y
    if 0 <= R1dot <= D && 0 <= R2dot <= D
        (x ⋅ y)::TensorForm{D,R1dot,R2dot}
        @test x ⋅ y === x ∨ ⋆y
    end

    norm2(x)::T
    @test norm2(x) >= 0
    @test norm2(a * x) == norm2(a) * norm2(x)
    # @test norm2(y + y2) <= norm2(y) + norm2(y2)
    norm(x)::float(T)
    @test norm(x) ≈ sqrt(norm2(x))
    @test norm(a * x) ≈ norm(a) * norm(x)
    @test norm(y + y2) <= norm(y) + norm(y2) || norm(y + y2) ≈ norm(y) + norm(y2)

    xs = rand(TensorForm{D,R1x,R2x,SVector{4,T}})
    xs2 = rand(TensorForm{D,R1x,R2x,SVector{4,T}})
    norm2(xs)::T
    @test norm2(xs) >= 0
    @test norm2(a * xs) == norm2(a) * norm2(xs)
    norm(xs)::float(T)
    @test norm(xs) ≈ sqrt(norm2(xs))
    @test norm(a * xs) ≈ norm(a) * norm(xs)
    @test norm(xs + xs2) <= norm(xs) + norm(xs2) || norm(xs + xs2) ≈ norm(xs) + norm(xs2)

    # cross product: x × y = ⋆(x ∧ y)
    R1cross = D - (R1x + R1y)
    R2cross = D - (R2x + R2y)
    if 0 <= R1cross && 0 <= R2cross
        (x × y)::TensorForm{D,R1cross,R2cross}
        @test x × y === ⋆(x ∧ y)
    end
end
