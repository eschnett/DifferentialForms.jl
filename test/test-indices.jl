using DifferentialForms
using DifferentialForms.Forms

using StaticArrays
using Test

@testset "binomial D=$D" for D in 0:10
    for R in 0:D
        @test binomial(Val(D), Val(R)) == binomial(D, R)
    end
end

@testset "bits2uint D=$D" for D in 0:10
    for uint in Unsigned(0):((Unsigned(1) << D) - 1)
        bits = DifferentialForms.Forms.uint2bits(Val(D), uint)
        uint′ = DifferentialForms.Forms.bits2uint(bits)
        @test uint′ == uint
    end
end

@testset "lin2bit D=$D" for D in 0:10
    for R in 0:D
        N = binomial(Val(D), Val(R))
        for lin in 1:N
            bits = DifferentialForms.Forms.lin2bit(Val(D), Val(R), lin)
            @test count(bits) == R
            lin′ = DifferentialForms.Forms.bit2lin(Val(D), Val(R), bits)
            @test lin′ == lin
        end
    end
end

@testset "bit2lst D=$D" for D in 0:10
    for R in 0:D
        N = binomial(Val(D), Val(R))
        for lin in 1:N
            bits = DifferentialForms.Forms.lin2bit(Val(D), Val(R), lin)
            lst = DifferentialForms.Forms.bit2lst(Val(D), Val(R), bits)
            @test length(lst) == R
            @test all(>=(1), lst)
            @test all(<=(D), lst)
            @test issorted(lst)
            @test length(unique(lst)) == R
            bits′ = DifferentialForms.Forms.lst2bit(Val(D), Val(R), lst)
            @test bits′ == bits
        end
    end
end

@testset "lin2lst D=$D" for D in 0:10
    for R in 0:D
        N = binomial(Val(D), Val(R))
        for lin in 1:N
            lst = DifferentialForms.Forms.lin2lst(Val(D), Val(R), lin)
            lin′ = DifferentialForms.Forms.lst2lin(Val(D), Val(R), lst)
            @test lin′ == lin
        end
    end
end
