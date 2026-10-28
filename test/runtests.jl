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
    # the AMPL starting point survives (elec_400.mod sets it with `let`)
    @test length(unique(mittelmann("elec_400").meta.x0)) == 1200

    for name in names
        # twod (126M variables, an 8.4 GiB file) is too large for the test run
        INFO[name]["nvar"] > 10^7 && continue
        get(INFO[name], "buffered", false) && !isdefined(ExaModels, :BufferedExpression) && continue
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
