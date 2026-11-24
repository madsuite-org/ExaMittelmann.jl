# Compiling Mittelmann instances into one shared library.
#
# Every instance is a fixed model read from its ExaModelsIO file, so its data is
# compiled into the library with it: the library grows with the instances in it, and
# `problems` picks them. Instances with buffered subexpressions (`WM_CFy`) are left out,
# since `juliac --trim` does not compile their Hessian.
module ExaMittelmannExaModelsCompilerExt

import ExaMittelmann
import ExaModelsCompiler
import ExaModelsIO
using ExaModelsCompiler: compile_library, select

# An instance name is not always a C identifier (`qcqp500-1c`); the prefix is the name in
# lowercase with every other character replaced by `_` (`narx_cfy`, `qcqp500_1c`).
_prefix(name::AbstractString) = Symbol(replace(lowercase(name), r"[^a-z0-9_]" => "_"))

_buffered(name) = get(ExaMittelmann.INSTANCES[name], "buffered", false)

"""
    compile_all(ExaMittelmann; path = "@mittelmann", problems, only = nothing,
                exclude = (), kwargs...)

Compile the instances named in `problems` into one shared library, each as a fixed
model under its name in lowercase with non-identifier characters replaced by `_`. The default is
the instances scored in the benchmark table with at most 100,000 variables, 41 of
the 47.
"""
function ExaModelsCompiler.compile_all(
    ::Val{ExaMittelmann};
    path = "@mittelmann",
    problems = [n for n in ExaMittelmann.mittelmann_instances(; benchmark = true)
                if ExaMittelmann.INSTANCES[n]["nvar"] <= 100_000 && !_buffered(n)],
    only = nothing,
    exclude = (),
    kwargs...,
)
    names = sort!(unique(String.(problems)))
    bad = filter(_buffered, names)
    isempty(bad) || throw(ArgumentError("instances with buffered subexpressions cannot be compiled: $(join(bad, ", "))"))
    pairs = [_prefix(n) => n for n in names]
    allunique(first.(pairs)) || throw(ArgumentError("two instances map to one prefix"))
    # Filter before reading: reading the files is the expensive part of assembling here.
    kept = Set(first.(select([p => nothing for p in first.(pairs)]; only, exclude)))
    models = [prefix => ExaModelsIO.read_core(ExaMittelmann.mittelmann_file(name))
              for (prefix, name) in pairs if prefix in kept]
    return compile_library(path, models...; kwargs...)
end

end # module
