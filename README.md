# ExaMittelmann.jl

The instances of Hans Mittelmann's [AMPL-NLP benchmark](https://plato.asu.edu/ftp/ampl-nlp.html)
as [ExaModels](https://github.com/madsuite-org/ExaModels.jl), loadable on CPU or GPU.

Each instance is stored as an [ExaModelsIO](https://github.com/madsuite-org/ExaModels.jl/tree/main/ExaModelsIO)
file (`.h5`) in a lazy Julia artifact, downloaded the first time it is used. All 123
models in the benchmark's source directory are included; 47 of them are the instances
scored in the benchmark table.

## Installation

ExaMittelmann and ExaModelsIO are not registered yet. Add ExaModelsIO first, then this
package:

```julia
using Pkg
Pkg.add(url = "https://github.com/madsuite-org/ExaModels.jl", subdir = "ExaModelsIO",
        rev = "58db52753d3a1f507389892b90649f80a74bfa77")
Pkg.add(url = "https://github.com/madsuite-org/ExaMittelmann.jl")
```

The instance files are attached to the `rolling` release, which follows `main` until the
package is registered.

```julia
using ExaMittelmann, CUDA

mittelmann_instances()                    # 123 instances
mittelmann_instances(; benchmark = true)  # the 47 scored in the benchmark table
mittelmann_file("clnlbeam")               # path of the .h5 file

m = mittelmann("clnlbeam")                         # CPU ExaModel
g = mittelmann("clnlbeam"; backend = CUDABackend()) # GPU ExaModel
```

Any NLPModels solver accepts these models, e.g. MadNLP:

```julia
using MadNLP, MadNLPGPU
madnlp(g)
```

## How the instances were built

The AMPL sources at `plato.asu.edu/ftp/ampl-nlp-source/` were translated to `.nl` by AMPL
(default options, so with AMPL's presolve, as in the benchmark runs), read as JuMP models,
and converted to ExaModels by `MittelmannHelper` (in `lib/MittelmannHelper`). Every instance
was checked against JuMP/MathOptInterface: objective, gradient, every constraint row and the
variable bounds agree to a relative 1e-8 at a perturbed starting point. It was then written
with `ExaModelsIO.write_model`, read back with `read_model`, and checked again: same sizes,
bounds and starting point, the same Jacobian and Hessian sparsity, and callback values that
agree to a relative 1e-12. `gen/generate.jl` rebuilds everything; it needs an AMPL license
without the demo size limit.

Two instances were written directly in ExaModels (`gen/twod.jl`, `gen/wm_cfy.jl`) and saved
the same way (`gen/extra.jl`):

- `twod` (126 million variables) is beyond what AMPL could translate in 62 GB. Its builder
  turns single-variable constraints into variable bounds, so it keeps the fixed and unused
  variables that AMPL's presolve removes. At `n = 20`, MadNLP reaches the same optimal value
  on it as on AMPL's `.nl` (relative difference 1.2e-11). Its file is 8.4 GiB.
- `WM_CFy`'s converted model, whose 1512 constraints each contain
  `sqrt((1/N) sum_i y[i,k]^2)`, did not compile. The builder makes the two norms buffered
  subexpressions; reading the file needs an ExaModels with buffered subexpressions (so far
  only the `compile-time-prototypes` branch). At a perturbed point, all 12850 constraint
  slacks of the model read from the file agree with AMPL's evaluation to a relative 1.3e-15.

ExaMittelmann needs ExaModelsIO 0.1.1 (madsuite-org/ExaModels.jl#350), which reads data
blocks with no columns (the constant terms of these models) and buffered subexpressions;
until it is registered, the commit above is used.

## Packages in this repository

- `ExaMittelmann` (this directory): the instances. `gen/` rebuilds them.
- `MittelmannHelper` (`lib/MittelmannHelper`): the pipeline that builds the files,
  `ampl_to_nl`, `read_nl`, `to_core`, `check_core`, `save_model` (ExaModelsIO
  `write_model`), `check_file` (reads the file back and compares) and
  `bind_file_artifact!`. ExaMittelmann does not depend on it.
