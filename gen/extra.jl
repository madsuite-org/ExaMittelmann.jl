# Write the two instances built directly in ExaModels as ExaModelsIO files.
#
#   julia --project=gen gen/extra.jl WORKDIR twod      # n = m = 500, as in twod.mod
#   julia --project=<env> gen/extra.jl WORKDIR WM_CFy  # needs buffered subexpressions
#
# twod's converted model has the size of twod.mod (126M variables), which AMPL could not
# translate in 62 GB; WM_CFy's converted model did not compile. Both builders were
# checked against AMPL: twod at n = 20 reaches the same optimum as AMPL's .nl (relative
# 1.2e-11), and every WM_CFy constraint slack agrees with AMPL's to a relative 1.3e-15.
# For WM_CFy, run with an environment whose ExaModels has `BufferedExpression` (branch
# compile-time-prototypes) and whose ExaModelsIO writes format 1.1.

using MittelmannHelper, TOML, Printf
import MittelmannHelper: ExaModels

include(joinpath(@__DIR__, "twod.jl"))
include(joinpath(@__DIR__, "wm_cfy.jl"))

function extra(workdir, name)
    out = mkpath(joinpath(workdir, "out"))
    m = name == "twod" ? twod() : name == "WM_CFy" ? wm_cfy() : error("unknown instance $name")
    file = save_model(joinpath(out, "$name.h5"), m)
    rt = check_file(file, m)
    rt.same_meta && rt.same_sparsity && maximum(rt.max_rel_diff) <= 1e-12 ||
        error("$name: the model read back from $file differs: $rt")
    info = Dict(
        "nvar" => m.meta.nvar, "ncon" => m.meta.ncon, "nnzj" => m.meta.nnzj, "nnzh" => m.meta.nnzh,
        "benchmark" => name == "WM_CFy", "source" => "https://plato.asu.edu/ftp/ampl-nlp-source/$name.mod",
        "builder" => "gen/$(lowercase(name)).jl",
        "check" => Dict("file" => Dict(string(k) => v for (k, v) in pairs(rt.max_rel_diff))),
    )
    name == "WM_CFy" && (info["buffered"] = true)
    open(io -> TOML.print(io, info), joinpath(out, "$name.toml"), "w")
    @printf("%-8s nvar %d ncon %d  file %.1e\n", name, m.meta.nvar, m.meta.ncon, maximum(rt.max_rel_diff))
end

extra(ARGS[1], ARGS[2])
