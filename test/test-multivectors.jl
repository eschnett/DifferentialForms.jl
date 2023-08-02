using DifferentialForms

using LinearAlgebra
using StaticArrays
using Test

@testset "Multivector masks and indices D=$D" for D in 0:Dmax
    Mbits = 2^D
    Mmax = UInt64(2)^Mbits - 1
    for iter in 1:100
        M = rand(1:Mmax)        # Don't test M == 0
        for iter2 in 1:100
            lin = rand(1:Multivectors.numelts(Val(D), Val(M)))
            bits = Multivectors.lin2bit(Val(D), Val(M), lin)
            @test Multivectors.bit2lin(Val(D), Val(M), bits) == lin
            lst = Multivectors.bit2lst(Val(D), bits)
            @test Multivectors.lst2bit(Val(D), lst) == bits
            @test Multivectors.lin2lst(Val(D), Val(M), lin) == lst
            @test Multivectors.lst2lin(Val(D), Val(M), lst) == lin
        end
    end
end

@testset "Create multivectors D=$D" for D in 0:Dmax
    γ = ntuple(d -> true, D)
    Mbits = 2^D
    Mmax = UInt64(2)^Mbits - 1

    T = BigRat

    for iter in 1:(100 ÷ (D + 1))
        M = rand(0:Mmax)
        N = length(Multivector{D,γ,M})
        x = Multivector{D,γ,M,T}(SVector{N,T}(rand(T) for n in 1:N))
        X = typeof(x)
        x′ = X(SVector{N,T}(rand(T) for n in 1:N))

        @test Multivector{D,γ,M,Float64}(x) == Multivector{D,γ,M,Float64}(x.elts)

        @test Multivector{D,γ,M,Float64}(collect(x)) == Multivector{D,γ,M,Float64}(x)
    end
end

@testset "Vector space operations on multivectors D=$D" for D in 0:Dmax
    γ = ntuple(d -> true, D)
    Mbits = 2^D
    Mmax = UInt64(2)^Mbits - 1

    T = Rational{Int64}

    for iter in 1:(100 ÷ (D + 1))
        Mn = zero(Mmax)
        Mx = rand(0:Mmax)
        My = rand(0:Mmax)
        Mz = rand(0:Mmax)
        n = zero(Multivector{D,γ,Mn,T})
        x = rand(Multivector{D,γ,Mx,T})
        y = rand(Multivector{D,γ,My,T})
        z = rand(Multivector{D,γ,Mz,T})
        a = rand(T)
        b = rand(T)

        # Vector space

        @test +x == x
        @test (x + y) + z == x + (y + z)

        @test n + x == x
        @test x + n == x

        @test x + y == y + x

        @test x + (-x) == n
        @test (-x) + x == n
        @test x - y == x + (-y)

        @test (a * b) * x == a * (b * x)
        @test x * (a * b) == (x * a) * b

        @test one(T) * a == a
        @test one(T) * x == x
        @test x * one(T) == x

        @test zero(T) * a == zero(T)
        @test zero(T) * x == n
        @test x * zero(T) == n

        if a != zero(T)
            @test a * inv(a) == one(T)

            @test inv(a) * (a * x) == x
            @test (x * a) * inv(a) == x

            @test inv(a) * x == a \ x
            @test x * inv(a) == x / a
        end

        @test (a + b) * x == a * x + b * x
        @test x * (a + b) == x * a + x * b

        @test a * (x + y) == a * x + a * y
        @test (x + y) * a == x * a + y * a

        # Nonlinear transformations
        @test map((a -> 2a), x) == 2 * x
        @test map((a, b) -> 2a + b, x, -x) == x
        @test map((a, b, c) -> 2a + b - c, x, x, 2 * x) == x
    end
end

@testset "Multivector algebra D=$D" for D in 0:Dmax
    γ = ntuple(d -> true, D)
    Mbits = 2^D
    Mmax = UInt64(2)^Mbits - 1

    T = Rational{Int64}

    for iter in 1:(100 ÷ (D + 1))
        Me = UInt64(0b1)
        Mx = rand(0:Mmax)
        My = rand(0:Mmax)
        Mz = rand(0:Mmax)
        e = one(Multivector{D,γ,Me,T})
        x = rand(Multivector{D,γ,Mx,T})
        y = rand(Multivector{D,γ,My,T})
        z = rand(Multivector{D,γ,Mz,T})
        a = rand(T)
        b = rand(T)

        # Multiplicative structure

        # units
        if D == 2
            e = unit(Multivector{D,γ,Me,T})
            e1 = unit(Multivector{D,γ,UInt64(0b0010),T}, 1)
            e2 = unit(Multivector{D,γ,UInt64(0b00100),T}, 2)
            e12 = unit(Multivector{D,γ,UInt64(0b1000),T}, 1, 2)

            @test ~e == e
            @test ~e1 == e1
            @test ~e2 == e2
            @test ~e12 == -e12

            @test ⋆e == e12
            @test ⋆e1 == e2
            @test ⋆e2 == -e1
            @test ⋆e12 == e

            @test e ∧ e == e
            @test e ∧ e1 == e1
            @test e ∧ e2 == e2
            @test e ∧ e12 == e12
            @test e1 ∧ e == e1
            @test e2 ∧ e == e2
            @test e12 ∧ e == e12
            @test e1 ∧ e1 == 0 * e12
            @test e1 ∧ e2 == e12
            @test e2 ∧ e1 == -e12
            @test e2 ∧ e2 == 0 * e12

            @test e ∨ e12 == e
            @test e1 ∨ e1 == 0 * e
            @test e2 ∨ e2 == 0 * e
            @test e1 ∨ e2 == e
            @test e1 ∨ e12 == e1
            @test e2 ∨ e12 == e2
            @test e12 ∨ e12 == e12

            @test e ⋅ e == e
            @test e1 ⋅ e1 == e
            @test e2 ⋅ e2 == e
            @test e1 ⋅ e2 == 0 * e
            @test e12 ⋅ e12 == e

            @test e × e == e12
            @test e × e1 == e2
            @test e × e2 == -e1
            @test e × e12 == e
            @test e1 × e1 == 0 * e
            @test e2 × e2 == 0 * e
            @test e1 × e2 == e

            @test e * e == e
            @test e * e1 == e1
            @test e * e2 == e2
            @test e * e12 == e12
            @test e1 * e == e1
            @test e2 * e == e2
            @test e12 * e == e12
            @test e1 * e1 == e
            @test e1 * e2 == e12
            @test e2 * e1 == -e12
            @test e2 * e2 == e
        end

        #TODO     # various duals
        #TODO     cycle_basis(x)::typeof(x)
        #TODO     x′ = x
        #TODO     for d in 1:D
        #TODO         x′ = cycle_basis(x′)
        #TODO     end
        #TODO     @test x′ == x
        #TODO     reverse_basis(x)::typeof(x)
        #TODO     @test reverse_basis(reverse_basis(x)) == x
        #TODO     @test reverse_basis(cycle_basis(reverse_basis(cycle_basis(x)))) == x

        @test ~~x == x
        @test a * ~x == ~(a * x)

        @test conj(conj(x)) == x
        @test a * conj(x) == conj(a * x)

        # @test ⋆⋆x == bitsign(R1 * (D - R1)) * x
        @test inv(⋆)(⋆x) == x
        @test ⋆inv(⋆)(x) == x
        @test ⋆⋆ ⋆ ⋆x == x
        @test a * ⋆x == ⋆(a * x)

        # exterior product: x ∧ y
        @test ∧(x) == x

        (x ∧ y)::Multivector{D}

        (x ∧ y ∧ z)::Multivector{D}
        @test (x ∧ y) ∧ z == x ∧ (y ∧ z)
        @test x ∧ y ∧ z == (x ∧ y) ∧ z
        @test ∧(x, y, z) == (x ∧ y) ∧ z
        @test ∧(SVector{0,typeof(x)}()) == e
        @test ∧(SVector(x)) == x
        @test ∧(SVector(x, y)) == x ∧ y
        @test ∧(SVector(x, y, z)) == (x ∧ y) ∧ z
        @test ∧(SVector{0,typeof(x)}()) ∧ x == x
        @test x ∧ ∧(SVector{0,typeof(x)}()) == x

        @test e ∧ x == x
        @test x ∧ e == x

        @test x ∧ zero(y) == zero(x ∧ y)
        @test zero(y) ∧ x == zero(y ∧ x)

        @test a * (x ∧ y) == x ∧ (a * y)
        @test x ∧ (y + z) == x ∧ y + x ∧ z

        @test x ∧ y == ~(~y ∧ ~x)

        # regressive product: ⋆(x ∨ y) = ⋆x ∧ ⋆y
        @test ∨(x) == x
        (x ∨ y)::Multivector{D}
        @test ⋆(x ∨ y) == ⋆x ∧ ⋆y

        (x ∨ y ∨ z)::Multivector{D}
        @test ⋆(x ∨ y ∨ z) == ⋆x ∧ ⋆y ∧ ⋆z
        @test x ∨ y ∨ z == (x ∨ y) ∨ z
        @test ∨(x, y, z) == (x ∨ y) ∨ z
        @test ∨(SVector{0,typeof(x)}()) == ⋆e
        @test ∨(SVector(x)) == x
        @test ∨(SVector(x, y)) == x ∨ y
        @test ∨(SVector(x, y, z)) == (x ∨ y) ∨ z
        @test ∨(SVector{0,typeof(x)}()) ∨ x == x
        @test x ∨ ∨(SVector{0,typeof(x)}()) == x

        @test ⋆e ∨ x == x
        @test x ∨ ⋆e == x

        @test x ∨ zero(y) == zero(x ∨ y)
        @test zero(y) ∨ x == zero(y ∨ x)

        @test a * (x ∨ y) == x ∨ (a * y)
        @test x ∨ (y + z) == (x ∨ y) + (x ∨ z)

        Q = inv(⋆) ∘ (~) ∘ (⋆)
        Q1 = (⋆) ∘ (~) ∘ inv(⋆)
        @test x ∨ y == Q1(Q(y) ∨ Q(x))

        # dot product: x ⋅ y = x ∨ ⋆y   (right contraction)
        (x ⋅ y)::Multivector{D}
        @test x ⋅ y == x ∨ ⋆y

        norm2(x)::T
        @test norm2(x) >= 0
        @test norm2(a * x) == norm2(a) * norm2(x)
        # @test norm2(y + y2) <= norm2(y) + norm2(y2)
        norm(x)::float(T)
        @test norm(x) ≈ sqrt(norm2(x))
        @test norm(a * x) ≈ norm(a) * norm(x)
        @test norm(x + y) <= norm(x) + norm(y) || norm(x + y) ≈ norm(x) + norm(y)

        # cross product: x × y = ⋆(x ∧ y)
        (x × y)::Multivector{D}
        @test x × y == ⋆(x ∧ y)

        # geometric product: x * y
        @test *(x) == x

        (x * y)::Multivector{D}

        (x * y * z)::Multivector{D}
        @test (x * y) * z == x * (y * z)
        @test x * y * z == (x * y) * z
        @test *(x, y, z) == (x * y) * z
        @test *(SVector{0,typeof(x)}()) == e
        @test *(SVector(x)) == x
        @test *(SVector(x, y)) == x * y
        @test *(SVector(x, y, z)) == (x * y) * z
        @test *(SVector{0,typeof(x)}()) * x == x
        @test x * *(SVector{0,typeof(x)}()) == x

        @test e * x == x
        @test x * e == x

        @test x * zero(y) == zero(x * y)
        @test zero(y) * x == zero(y * x)

        @test a * (x * y) == x * (a * y)
        @test x * (y + z) == x * y + x * z
    end
end

@testset "Multivector algebra with arbitrary metrics D=$D" for D in 0:Dmax
    Mbits = 2^D
    Mmax = UInt64(2)^Mbits - 1

    T = Rational{Int64}

    for iter in 1:(100 ÷ (D + 1))
        γ = NTuple{D,Int8}(rand(-1:1, D))

        Me = UInt64(0b1)
        Mx = rand(0:Mmax)
        My = rand(0:Mmax)
        Mz = rand(0:Mmax)
        e = one(Multivector{D,γ,Me,T})
        x = rand(Multivector{D,γ,Mx,T})
        y = rand(Multivector{D,γ,My,T})
        z = rand(Multivector{D,γ,Mz,T})
        a = rand(T)
        b = rand(T)

        # Multiplicative structure

        # units
        if D == 2
            e = unit(Multivector{D,γ,Me,T})
            e1 = unit(Multivector{D,γ,UInt64(0b0010),T}, 1)
            e2 = unit(Multivector{D,γ,UInt64(0b00100),T}, 2)
            e12 = unit(Multivector{D,γ,UInt64(0b1000),T}, 1, 2)

            @test e * e == e
            @test e1 * e1 == γ[1] * e
            @test e2 * e2 == γ[2] * e
            @test e * e12 == e12
            @test e1 * e12 == γ[1] * e2
            @test e2 * e12 == -γ[2] * e1
            @test e12 * e1 == -γ[1] * e2
            @test e12 * e2 == γ[2] * e1
            @test e12 * e12 == -γ[1] * γ[2] * e
            @test e12 * ~e12 == γ[1] * γ[2] * e
        end
    end
end

#TODO const Dmax4 = min(4, Dmax)
#TODO @testset "Tensor sums of multivectors D1=$D1 D2=$D2 R=$R" for D1 in 0:Dmax4,
#TODO D2 in 0:Dmax4,
#TODO R in 1:min(D1, D2)
#TODO 
#TODO     γ = ntuple(d->true, D)
#TODO 
#TODO     D = D1 + D2
#TODO 
#TODO     T = Rational{Int64}
#TODO     x = rand(Form{D1,R,T})
#TODO     x2 = rand(Form{D1,R,T})
#TODO     y = rand(Form{D2,R,T})
#TODO     y2 = rand(Form{D2,R,T})
#TODO     a = rand(T)
#TODO 
#TODO     # units
#TODO     if D1 == 1 && D2 == 1 && R == 1
#TODO         u(d, inds...) = unit(Form{d,length(inds),T}, inds...)
#TODO 
#TODO         @test u(1, 1) ⊕ u(1, 1) == u(2, 1) + u(2, 2)
#TODO 
#TODO         @test u(1, 1) ⊕ u(2, 1) == u(3, 1) + u(3, 2)
#TODO         @test u(1, 1) ⊕ u(2, 2) == u(3, 1) + u(3, 3)
#TODO 
#TODO         @test u(3, 1) ⊕ u(2, 1) == u(5, 1) + u(5, 4)
#TODO         @test u(3, 1) ⊕ u(2, 2) == u(5, 1) + u(5, 5)
#TODO         @test u(3, 2) ⊕ u(2, 1) == u(5, 2) + u(5, 4)
#TODO         @test u(3, 2) ⊕ u(2, 2) == u(5, 2) + u(5, 5)
#TODO         @test u(3, 3) ⊕ u(2, 1) == u(5, 3) + u(5, 4)
#TODO         @test u(3, 3) ⊕ u(2, 2) == u(5, 3) + u(5, 5)
#TODO     end
#TODO 
#TODO     @test ⊕(x) == x
#TODO     @test a * ⊕(x) == ⊕(a * x)
#TODO 
#TODO     (x ⊕ y)::Form{D,R,T}
#TODO     @test a * (x ⊕ y) == (a * x) ⊕ (a * y)
#TODO     @test (x ⊕ y) * a == (x * a) ⊕ (y * a)
#TODO     @test x ⊕ y == (zero(x) ⊕ y) + (x ⊕ zero(y))
#TODO     @test (x + x2) ⊕ (y + y2) == (x ⊕ y) + (x2 ⊕ y2)
#TODO 
#TODO     @test reverse_basis(x ⊕ y) ==
#TODO           # bitsign(R * R) * 
#TODO           (reverse_basis(y) ⊕ reverse_basis(x))
#TODO 
#TODO     for D3 in R:Dmax4
#TODO         z = rand(Form{D3,R,T})
#TODO         @test (x ⊕ y) ⊕ z == x ⊕ (y ⊕ z)
#TODO         @test x ⊕ y ⊕ z == x ⊕ (y ⊕ z)
#TODO         @test ⊕(x, y, z) == x ⊕ (y ⊕ z)
#TODO     end
#TODO end
#TODO 
#TODO const Dmax3 = min(3, Dmax)
#TODO @testset "Tensor products of multivectors D1=$D1 R1=$R1 D2=$D2 R2=$R2" for D1 in
#TODO                                                                            0:Dmax3,
#TODO D2 in 0:Dmax3,
#TODO R1 in 0:D1,
#TODO R2 in 0:D2
#TODO 
#TODO     γ = ntuple(d->true, D)
#TODO 
#TODO     D = D1 + D2
#TODO     R = R1 + R2
#TODO 
#TODO     T = Rational{Int64}
#TODO     e = one(Form{0,0,T})
#TODO     x = rand(Form{D1,R1,T})
#TODO     x2 = rand(Form{D1,R1,T})
#TODO     y = rand(Form{D2,R2,T})
#TODO     y2 = rand(Form{D2,R2,T})
#TODO     a = rand(T)
#TODO 
#TODO     # units
#TODO     if D1 == 1 && D2 == 1 && R1 == 1 && R2 == 1
#TODO         u(d, inds...) = unit(Form{d,length(inds),T}, inds...)
#TODO 
#TODO         @test u(1) ⊗ u(1) == u(2)
#TODO         @test u(1, 1) ⊗ u(1) == u(2, 1)
#TODO         @test u(1) ⊗ u(1, 1) == u(2, 2)
#TODO         @test u(1, 1) ⊗ u(1, 1) == u(2, 1, 2)
#TODO 
#TODO         @test u(1) ⊗ u(2) == u(3)
#TODO         @test u(1, 1) ⊗ u(2) == u(3, 1)
#TODO         @test u(1) ⊗ u(2, 1) == u(3, 2)
#TODO         @test u(1, 1) ⊗ u(2, 1) == u(3, 1, 2)
#TODO         @test u(1) ⊗ u(2, 2) == u(3, 3)
#TODO         @test u(1, 1) ⊗ u(2, 2) == u(3, 1, 3)
#TODO         @test u(1) ⊗ u(2, 1, 2) == u(3, 2, 3)
#TODO         @test u(1, 1) ⊗ u(2, 1, 2) == u(3, 1, 2, 3)
#TODO     end
#TODO 
#TODO     @test ⊗(x) == x
#TODO     @test a * ⊗(x) == ⊗(a * x)
#TODO 
#TODO     (x ⊗ y)::Form{D,R,T}
#TODO     @test a * (x ⊗ y) == (a * x) ⊗ y
#TODO     @test (x ⊗ y) * a == x ⊗ (y * a)
#TODO     @test e ⊗ y == y
#TODO     @test x ⊗ e == x
#TODO     @test (x + x2) ⊗ y == x ⊗ y + x2 ⊗ y
#TODO     @test x ⊗ (y + y2) == x ⊗ y + x ⊗ y2
#TODO 
#TODO     @test reverse_basis(x ⊗ y) ==
#TODO           bitsign(R1 * R2) * (reverse_basis(y) ⊗ reverse_basis(x))
#TODO 
#TODO     for D3 in 0:Dmax3, R3 in 0:D3
#TODO         z = rand(Form{D3,R3,T})
#TODO         @test (x ⊗ y) ⊗ z == x ⊗ (y ⊗ z)
#TODO         @test x ⊗ y ⊗ z == x ⊗ (y ⊗ z)
#TODO         @test ⊗(x, y, z) == x ⊗ (y ⊗ z)
#TODO     end
#TODO end
