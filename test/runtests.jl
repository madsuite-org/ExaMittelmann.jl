using ExaMittelmann, NLPModels, Test
import ExaModels, TOML

const INFO = TOML.parsefile(joinpath(pkgdir(ExaMittelmann), "instances.toml"))

const GPU = try
    @eval using CUDA, KernelAbstractions
    CUDA.functional()
catch
    false
end

function callbacks(m, x, y)
    return (obj(m, x), Array(grad(m, x)), Array(cons(m, x)), Array(jac_coord(m, x)),
            Array(hess_coord(m, x, y)))
end

@testset "ExaMittelmann" begin
    names = mittelmann_instances()
    @test issubset(mittelmann_instances(; benchmark = true), names)
    @test_throws ArgumentError mittelmann_file("no_such_instance")

    @testset "twod" begin
        for (n, m) in ((10, 10), (12, 7))
            t = twod(; n, m)
            @test t.meta.nvar == (m + 1) * (n + 1)^2 + m * (n - 1)
            @test t.meta.ncon == m * (n - 1)^2 + 4 * m * (n - 1)
            @test isfinite(obj(t, t.meta.x0))
            if GPU
                g = twod(; n, m, backend = CUDABackend())
                x = t.meta.x0 .+ 1e-3 .* sin.(1:t.meta.nvar)
                @test obj(g, CuArray(x)) ≈ obj(t, x) rtol = 1e-12
                @test Array(cons(g, CuArray(x))) ≈ cons(t, x) rtol = 1e-12
            end
        end
    end

    @testset "wm_cfy" begin
        if isdefined(ExaModels, :BufferedExpression)
            w = wm_cfy()
            @test (w.meta.nvar, w.meta.ncon) == (8711, 12850)
            @test isfinite(obj(w, w.meta.x0))
        else
            @test_throws ErrorException wm_cfy()
        end
    end

    for name in names
        @testset "$name" begin
            m = mittelmann(name)
            @test m isa ExaModels.ExaModel
            @test m.meta.nvar == INFO[name]["nvar"]
            @test m.meta.ncon == INFO[name]["ncon"]
            # near the starting point, inside the bounds (some instances are singular at
            # x0 itself, e.g. log of a variable that starts at its lower bound 0)
            x0 = m.meta.x0
            x = clamp.(x0 .+ 1e-3 .* (1 .+ abs.(x0)) .* abs.(sin.(1:m.meta.nvar)),
                       m.meta.lvar, m.meta.uvar)
            y = cos.(1:m.meta.ncon)
            ref = callbacks(m, x, y)
            @test all(r -> all(isfinite, r), ref)
            if GPU
                g = mittelmann(name; backend = CUDABackend())
                @test g.meta.x0 isa CuArray
                res = callbacks(g, CuArray(x), CuArray(y))
                # the objective is a sum reduced in a different order on the device; in
                # lukvle7/lukvli7 its 250000 terms cancel by six orders of magnitude
                @test isapprox(res[1], ref[1]; rtol = 1e-6, nans = true)
                for (a, b) in zip(res[2:end], ref[2:end])
                    @test isapprox(a, b; rtol = 1e-8, nans = true)
                end
            end
        end
    end
end
