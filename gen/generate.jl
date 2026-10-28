# Build the ExaMittelmann artifacts from the AMPL sources.
#
#   julia --project=gen gen/generate.jl list WORKDIR
#   AMPL=/path/to/ampl julia --project=gen gen/generate.jl build WORKDIR name
#   [ARTIFACT_URL=https://...] julia --project=gen gen/generate.jl bind WORKDIR
#
# `build` takes one model from plato.asu.edu/ftp/ampl-nlp-source/ through
#   .mod --AMPL--> .nl --JuMP--> ExaModel --ExaModelsIO--> WORKDIR/out/<name>.h5,
# checks the ExaModel against JuMP/MOI and the file read back against the ExaModel, and
# writes WORKDIR/out/<name>.toml. Builds are independent, so run them in parallel
# (`list WORKDIR | xargs -P`). `bind` turns every instance's .h5 into a lazy artifact in
# Artifacts.toml, writes instances.toml, and writes the tarballs to upload to
# WORKDIR/tarballs.
# Needs an unlimited AMPL license (the demo caps nonlinear models at 300 vars/cons).

using MittelmannHelper, JuMP, TOML, Downloads, Printf
import MittelmannHelper: ExaModels

const SOURCE = "https://plato.asu.edu/ftp/ampl-nlp-source/"
const PKG = dirname(@__DIR__)

# Instances scored at plato.asu.edu/ftp/ampl-nlp.html (9 Sep 2026), by source file name.
const BENCHMARK = Set([
    "arki0003", "arki0009", "bearing_400", "camshape_6400", "clnlbeam", "cont5_1_l",
    "cont5_2_1_l", "cont5_2_2_l", "cont5_2_3_l", "cont5_2_4_l", "corkscrw", "dirichlet120",
    "dtoc1nd", "dtoc2", "elec_400", "ex1_160", "ex1_320", "ex4_2_160", "ex4_2_320", "ex8_2_2",
    "ex8_2_3", "gasoil_3200", "henon120", "lane_emden120", "marine_1600", "NARX_CFy", "nql180",
    "optmass", "pinene_3200", "qcqp500-3c", "qcqp500-3nc", "qcqp750-2c", "qcqp750-2nc",
    "qcqp1000-1nc", "qcqp1000-2c", "qcqp1000-2nc", "qcqp1500-1c", "qcqp1500-1nc", "qssp180",
    "robot_1600", "robot_a", "robot_b", "robot_c", "rocket_12800", "steering_12800",
    "svanberg", "WM_CFy",
])

function source_names()
    html = String(take!(Downloads.download(SOURCE, IOBuffer())))
    files = unique([m.captures[1] for m in eachmatch(r"href=\"([^\"/?]+\.mod(?:\.gz)?)\"", html)])
    return sort!([replace(f, r"\.mod(\.gz)?$" => "") => f for f in files])
end

function fetch_mod(dir, name, file)
    mod = joinpath(dir, "$name.mod")
    isfile(mod) && return mod
    Downloads.download(SOURCE * file, joinpath(dir, file))
    endswith(file, ".gz") && run(`gunzip -f $(joinpath(dir, file))`)
    return mod
end

function build(workdir, name)
    # the source listing is fetched once by `list`, not by each parallel build
    file_src = TOML.parsefile(joinpath(workdir, "sources.toml"))[name]
    moddir = mkpath(joinpath(workdir, "mod"))
    out = mkpath(joinpath(workdir, "out"))
    nl = joinpath(moddir, "$name.nl")
    isfile(nl) || ampl_to_nl(fetch_mod(moddir, name, file_src); ampl = get(ENV, "AMPL", "ampl"))
    t = @elapsed begin
        jm = read_nl(nl)
        core, maps = to_core(jm)
        err = check_core(core, maps, jm)
    end
    maximum(err) < 1e-8 || error("$name: ExaModel disagrees with JuMP: $err")
    m = ExaModels.ExaModel(core)
    file = save_model(joinpath(out, "$name.h5"), m)
    rt = check_file(file, m)
    rt.same_meta && rt.same_sparsity && maximum(rt.max_rel_diff) <= 1e-12 ||
        error("$name: the model read back from $file differs: $rt")
    info = Dict(
        "nvar" => core.nvar, "ncon" => core.ncon, "nnzj" => core.nnzj, "nnzh" => core.nnzh,
        "benchmark" => name in BENCHMARK, "source" => SOURCE * file_src,
        "check" => Dict("jump" => Dict(string(k) => v for (k, v) in pairs(err)),
                        "file" => Dict(string(k) => v for (k, v) in pairs(rt.max_rel_diff))),
    )
    open(io -> TOML.print(io, info), joinpath(out, "$name.toml"), "w")
    @printf("%-16s nvar %7d ncon %7d  jump %.1e  file %.1e  %.0fs\n", name, core.nvar, core.ncon,
            maximum(err), maximum(rt.max_rel_diff), t)
end

function bind(workdir)
    out = joinpath(workdir, "out")
    url = get(ENV, "ARTIFACT_URL", nothing)       # base URL the tarballs will be served from
    toml = joinpath(PKG, "Artifacts.toml")
    info = Dict{String, Any}()
    for f in sort(readdir(out))
        endswith(f, ".toml") || continue
        name = f[1:end-5]
        meta = TOML.parsefile(joinpath(out, f))
        delete!(meta, "check")
        # resumable: an instance already packed and bound is not packed again
        done = isfile(joinpath(workdir, "tarballs", "$name.tar.gz")) && isfile(toml) &&
               haskey(TOML.parsefile(toml), name)
        done || bind_file_artifact!(toml, name, joinpath(out, "$name.h5");
                                    tarballs = joinpath(workdir, "tarballs"), url)
        info[name] = meta
    end
    open(io -> TOML.print(io, info; sorted = true), joinpath(PKG, "instances.toml"), "w")
    println("bound $(length(info)) instances")
end

function list(workdir)
    src = Dict(source_names())
    open(io -> TOML.print(io, src; sorted = true), joinpath(mkpath(workdir), "sources.toml"), "w")
    foreach(println, sort!(collect(keys(src))))
end

cmd = ARGS[1]
cmd == "list" ? list(ARGS[2]) :
cmd == "build" ? build(ARGS[2], ARGS[3]) :
cmd == "bind" ? bind(ARGS[2]) : error("unknown command $cmd")
