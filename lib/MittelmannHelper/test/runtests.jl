using MittelmannHelper, JuMP, NLPModels, KernelAbstractions, Test
import ExaModels
const ExaModelsIO = MittelmannHelper.ExaModelsIO

const GPU = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

callbacks(m, x, y) = (obj(m, x), Array(grad(m, x)), Array(cons(m, x)),
                      Array(jac_coord(m, x)), Array(hess_coord(m, x, y)))

function jump_model()
    m = Model()
    @variable(m, -1 <= t[1:20] <= 1, start = 0.05)
    @variable(m, x[1:20] >= -0.05, start = 0.01)
    @variable(m, u[1:20])
    fix(x[1], 0.0; force = true)
    @objective(m, Min, sum(0.5 * u[i]^2 + 0.5 * cos(t[i]) for i in 1:20) + exp(2.0))
    @constraint(m, [i in 1:19], x[i+1] - x[i] - 0.05 * (sin(t[i+1]) + sin(t[i])) == 0)
    @constraint(m, [i in 1:19], t[i+1] - t[i] - 0.05 * u[i+1] - 0.05 * u[i] == 0)
    @constraint(m, [i in 1:19], -1 <= u[i] * t[i] <= 2)
    @constraint(m, sum(x) >= -1)
    return m
end

@testset "MittelmannHelper" begin
    jm = jump_model()
    core, maps = to_core(jm)
    err = check_core(core, maps, jm)
    @test maximum(err) < 1e-12

    dir = mktempdir()
    m1 = ExaModels.ExaModel(core)
    file = save_model(joinpath(dir, "m.h5"), core)
    chk = check_file(file, m1)
    @test chk.same_meta && chk.same_sparsity
    @test maximum(chk.max_rel_diff) <= 1e-12
    m2 = ExaModelsIO.read_model(file)
    x = m1.meta.x0 .+ 1e-2 .* sin.(1:m1.meta.nvar)
    @test obj(m2, x) ≈ obj(m1, x) rtol = 1e-14
    @test cons(m2, x) ≈ cons(m1, x) rtol = 1e-14

    # the .nl route: write the JuMP model to .nl and read it back
    nl = joinpath(dir, "m.nl")
    write_to_file(jm, nl)
    jn = read_nl(nl)
    cn, mn = to_core(jn)
    @test maximum(check_core(cn, mn, jn)) < 1e-12
    @test ExaModels.ExaModel(cn).meta.ncon == m1.meta.ncon

    if GPU
        g = ExaModelsIO.read_model(file; backend = CUDABackend())
        @test g.meta.x0 isa CuArray
        y = cos.(1:m1.meta.ncon)
        for (a, b) in zip(callbacks(g, CuArray(x), CuArray(y)), callbacks(m2, x, y))
            @test a ≈ b rtol = 1e-12
        end
    end

    toml = joinpath(dir, "Artifacts.toml")
    h = bind_file_artifact!(toml, "m", file; tarballs = joinpath(dir, "tar"),
                            url = "https://example.org/a")
    @test isfile(joinpath(dir, "tar", "m.tar.gz"))
    @test occursin("https://example.org/a/m.tar.gz", read(toml, String))
    @test isfile(joinpath(MittelmannHelper.Artifacts.artifact_path(h), "m.h5"))

    if haskey(ENV, "AMPL")
        mod = joinpath(dir, "small.mod")
        write(mod, """
        param n := 10;
        var x{1..n} := 0.5;
        minimize f: sum{i in 1..n} (x[i] - 1)^4 + 0.5 * sin(x[1]);
        s.t. c{i in 1..n-1}: x[i]^2 + x[i+1] <= 2;
        solve;
        display x;
        """)
        jn = read_nl(ampl_to_nl(mod; ampl = ENV["AMPL"]))
        c, mp = to_core(jn)
        @test maximum(check_core(c, mp, jn)) < 1e-12
    end
end
