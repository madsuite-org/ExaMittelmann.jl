"""
    ExaMittelmann

The Mittelmann AMPL-NLP benchmark instances (plato.asu.edu/ftp/ampl-nlp.html) as
`ExaModels`, stored as ExaModelsIO files in artifacts and loadable on CPU or GPU.
"""
module ExaMittelmann

using Artifacts, LazyArtifacts, TOML
import ExaModels, ExaModelsIO

export mittelmann, mittelmann_file, mittelmann_instances

const ARTIFACTS_TOML = joinpath(@__DIR__, "..", "Artifacts.toml")
const INSTANCES = TOML.parsefile(joinpath(@__DIR__, "..", "instances.toml"))

"""
    mittelmann_instances(; benchmark = false)

Names of the available instances, sorted. With `benchmark = true`, only the
instances scored in the Mittelmann AMPL-NLP benchmark.
"""
mittelmann_instances(; benchmark = false) =
    sort!([k for (k, v) in INSTANCES if !benchmark || v["benchmark"]])

"""
    mittelmann_file(name)

Path of the ExaModelsIO file (`.h5`) of instance `name`, downloading its artifact on
first use.
"""
function mittelmann_file(name::AbstractString)
    haskey(INSTANCES, name) || throw(ArgumentError("unknown Mittelmann instance $name"))
    dir = LazyArtifacts.ensure_artifact_installed(name, ARTIFACTS_TOML)
    return joinpath(dir, "$name.h5")
end

"""
    mittelmann(name; backend = nothing)

Instance `name` as an `ExaModel` on `backend` (`nothing` for CPU, or a
KernelAbstractions backend such as `CUDABackend()`), read with
`ExaModelsIO.read_model`. `WM_CFy` needs an ExaModels with buffered subexpressions.
"""
function mittelmann(name::AbstractString; backend = nothing)
    file = mittelmann_file(name)
    if get(INSTANCES[name], "buffered", false) && !isdefined(ExaModels, :BufferedExpression)
        error("$name uses buffered subexpressions, which ExaModels $(pkgversion(ExaModels)) " *
              "does not have; they are on the ExaModels branch compile-time-prototypes")
    end
    return ExaModelsIO.read_model(file; backend)
end

end # module ExaMittelmann
