using DifferentialForms

using StaticArrays
using Test

@testset "bitsign" begin
    for b in false:true
        @test abs(bitsign(b)) === 1
        @test signbit(bitsign(b)) === b
        @test bitsign(Int(b)) === bitsign(b)
    end
    for n in 1:100
        i = Int(rand(Int8))
        j = Int(rand(Int8))
        @test bitsign(i) * bitsign(j) == bitsign(i + j)
    end
end

@testset "sort_perm (StaticArray)" begin
    @test sort_perm(SVector{0,Int}()) == (SVector{0,Int}(), 0)
    @test sort_perm(SVector(1)) == (SVector(1), 0)
    @test sort_perm(SVector(1, 2)) == (SVector(1, 2), 0)
    @test sort_perm(SVector(2, 1)) == (SVector(1, 2), 1)
    @test sort_perm(SVector(1, 2, 3)) == (SVector(1, 2, 3), 0)
    @test sort_perm(SVector(1, 3, 2)) == (SVector(1, 2, 3), 1)
    @test sort_perm(SVector(2, 1, 3)) == (SVector(1, 2, 3), 1)
    @test sort_perm(SVector(2, 3, 1)) == (SVector(1, 2, 3), 2)
    @test sort_perm(SVector(3, 1, 2)) == (SVector(1, 2, 3), 2)
    @test sort_perm(SVector(3, 2, 1)) == (SVector(1, 2, 3), 3)

    for N in 0:10
        xs = rand(SVector{N,Int})
        ys, s = sort_perm(xs)
        @test issorted(ys)
        @test s >= 0
    end

    for N in 2:10
        xs = SVector{N}(1:N)
        n = rand(0:10)
        # permute n times
        for i in 1:n
            j = rand(1:(N - 1))
            xsj = xs[j]
            xsj1 = xs[j + 1]
            xs = setindex(xs, xsj1, j)
            xs = setindex(xs, xsj, j + 1)
        end
        ys, s = sort_perm(xs)
        @test issorted(ys)
        @test iseven(s - n)
    end
end

@testset "sort_perm (AbstractVector)" begin
    @test sort_perm(Vector{Int}()) == (Vector{Int}(), 0)
    @test sort_perm([1]) == ([1], 0)
    @test sort_perm([1, 2]) == ([1, 2], 0)
    @test sort_perm([2, 1]) == ([1, 2], 1)
    @test sort_perm([1, 2, 3]) == ([1, 2, 3], 0)
    @test sort_perm([1, 3, 2]) == ([1, 2, 3], 1)
    @test sort_perm([2, 1, 3]) == ([1, 2, 3], 1)
    @test sort_perm([2, 3, 1]) == ([1, 2, 3], 2)
    @test sort_perm([3, 1, 2]) == ([1, 2, 3], 2)
    @test sort_perm([3, 2, 1]) == ([1, 2, 3], 3)

    for N in 0:10
        xs = rand(Int, N)
        ys, s = sort_perm(xs)
        @test issorted(ys)
        @test s >= 0
    end

    for N in 2:10
        xs = collect(1:N)
        n = rand(0:10)
        # permute n times
        for i in 1:n
            j = rand(1:(N - 1))
            xs[j], xs[j + 1] = xs[j + 1], xs[j]
        end
        ys, s = sort_perm(xs)
        @test issorted(ys)
        @test iseven(s - n)
    end
end
