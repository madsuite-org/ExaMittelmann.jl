"""
    MittelmannHelper

Turn AMPL models into ExaModelsIO files.

Pipeline: an AMPL `.mod` file is translated to `.nl` ([`ampl_to_nl`](@ref)), read as a
JuMP model ([`read_nl`](@ref)), converted to a CPU `ExaCore` ([`to_core`](@ref)) and
checked against JuMP ([`check_core`](@ref)). [`save_model`](@ref) writes it with
`ExaModelsIO.write_model`, and [`check_file`](@ref) reads the file back and compares
it with the model it came from. [`bind_file_artifact!`](@ref) packages a file as a
Julia artifact. Read the files with `ExaModelsIO.read_model(path; backend)`.

`read_nl`, `to_core` and `check_core` need `JuMP` to be loaded.
"""
module MittelmannHelper

import ExaModels, ExaModelsIO
import ExaModels: NLPModels
import Pkg.Artifacts
using SparseArrays

export ampl_to_nl, read_nl, to_core, check_core
export save_model, check_file, bind_file_artifact!

"""
    save_model(file, core_or_model)

Write an `ExaCore` or `ExaModel` to `file` with `ExaModelsIO.write_model`.
"""
save_model(file::AbstractString, core::ExaModels.ExaCore) = save_model(file, ExaModels.ExaModel(core))
function save_model(file::AbstractString, m::ExaModels.ExaModel)
    ExaModelsIO.write_model(file, m)
    return file
end

# A point near the starting point, inside the bounds (some models are singular at
# the starting point itself, e.g. log of a variable that starts at its lower bound).
function _test_point(m)
    x0 = m.meta.x0
    return clamp.(x0 .+ 1e-3 .* (1 .+ abs.(x0)) .* abs.(sin.(1:m.meta.nvar)), m.meta.lvar, m.meta.uvar)
end

# relative error; entries equal as values (including matching infinities) count as 0
_rel(a, b) = maximum((p == q ? 0.0 : abs(p - q) / (1 + abs(q)) for (p, q) in zip(a, b)); init = 0.0)

"""
    check_file(file, m::ExaModel) -> NamedTuple

Read `file` with `ExaModelsIO.read_model` and compare it with `m`: sizes, bounds and
starting point (`same_meta`), Jacobian and Hessian sparsity (`same_sparsity`), and the
largest relative difference of each callback (`max_rel_diff`) at a point near the
start. `read_model` rebuilds the model, so Jacobian and Hessian coordinates may come
in another order; they are compared as sparse matrices with duplicates summed.
"""
function check_file(file::AbstractString, m::ExaModels.ExaModel)
    r = ExaModelsIO.read_model(file)
    F = NLPModels
    x = _test_point(m)
    y = cos.(1:m.meta.ncon)
    same_meta = all(getfield(r.meta, f) == getfield(m.meta, f)
                    for f in (:nvar, :ncon, :nnzj, :nnzh, :x0, :lvar, :uvar, :y0, :lcon, :ucon, :minimize))
    J(mod) = (s = F.jac_structure(mod); sparse(s[1], s[2], F.jac_coord(mod, x), mod.meta.ncon, mod.meta.nvar))
    H(mod) = (s = F.hess_structure(mod); sparse(s[1], s[2], F.hess_coord(mod, x, y), mod.meta.nvar, mod.meta.nvar))
    Jr, Jm, Hr, Hm = J(r), J(m), H(r), H(m)
    same_sparsity = Jr.colptr == Jm.colptr && Jr.rowval == Jm.rowval &&
                    Hr.colptr == Hm.colptr && Hr.rowval == Hm.rowval
    max_rel_diff = (
        obj = _rel([F.obj(r, x)], [F.obj(m, x)]),
        grad = _rel(F.grad(r, x), F.grad(m, x)),
        cons = _rel(F.cons(r, x), F.cons(m, x)),
        jac = same_sparsity ? _rel(Jr.nzval, Jm.nzval) : Inf,
        hess = same_sparsity ? _rel(Hr.nzval, Hm.nzval) : Inf,
    )
    return (; same_meta, same_sparsity, max_rel_diff)
end

"""
    ampl_to_nl(mod; ampl = "ampl", dir = dirname(mod), drop = r"^\\s*(solve|display)\\b")

Translate the AMPL model `mod` to `<dir>/<name>.nl` with the AMPL executable `ampl`
and return the path. Lines matching `drop` are removed first, so a model file that
also calls a solver or prints can still be translated. AMPL's output goes to
`<dir>/<name>.ampl.log`. Models above 300 nonlinear variables or constraints need a
full AMPL license (the free Community Edition has no size limit).
"""
function ampl_to_nl(mod::AbstractString; ampl = "ampl", dir = dirname(abspath(mod)),
                    drop = r"^\s*(solve|display)\b")
    name = replace(basename(mod), r"\.mod$" => "")
    src = joinpath(dir, "$name.clean.mod")
    write(src, join(filter(l -> !occursin(drop, l), readlines(mod)), "\n"), "\n")
    nl = joinpath(dir, "$name.nl")
    rm(nl; force = true)
    log = joinpath(dir, "$name.ampl.log")
    cd(dir) do
        run(pipeline(ignorestatus(`$ampl -og$name $src`); stdout = log, stderr = log))
    end
    isfile(nl) || error("AMPL wrote no $nl:\n" * read(log, String))
    return nl
end

"""
    bind_file_artifact!(artifacts_toml, name, file; tarballs, url = nothing, lazy = true)

Put `file` (e.g. an ExaModelsIO `.h5`) in a new artifact, bind it as `name` in
`artifacts_toml`, and write `<tarballs>/<name>.tar.gz`. With `url`, the binding
downloads from `<url>/<name>.tar.gz`. Returns the artifact hash.
"""
function bind_file_artifact!(artifacts_toml, name, file; tarballs, url = nothing, lazy = true)
    hash = Artifacts.create_artifact() do dir
        cp(file, joinpath(dir, basename(file)))
    end
    sha = Artifacts.archive_artifact(hash, joinpath(mkpath(tarballs), "$name.tar.gz"))
    dl = url === nothing ? nothing : [("$url/$name.tar.gz", sha)]
    Artifacts.bind_artifact!(artifacts_toml, name, hash; download_info = dl, lazy, force = true)
    return hash
end

# Implemented in the JuMP extension.
"""
    read_nl(file) -> JuMP.Model

Read an `.nl` file as a JuMP model whose nonlinear parts are `ScalarNonlinearFunction`s,
the form the ExaModels converter accepts. Needs `JuMP`.
"""
function read_nl end

"""
    to_core(model::JuMP.Model) -> (core, (var_to_idx, con_to_idx))

Convert a JuMP model to a CPU `ExaCore`. The maps send each MOI variable and
constraint index to its position in the `ExaModel`, which groups constraints by
type and so may order them differently from JuMP. Needs `JuMP`.
"""
function to_core end

"""
    check_core(core, maps, model::JuMP.Model; x = nothing) -> NamedTuple

Evaluate the `ExaModel` of `core` and the JuMP `model` at `x` (default: the starting
point plus a small perturbation) and return the largest relative differences in
objective, gradient, constraint rows and variable bounds. Needs `JuMP`.
"""
function check_core end

end # module MittelmannHelper
