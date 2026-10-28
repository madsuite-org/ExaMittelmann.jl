# ExaMittelmann.jl

The instances of Hans Mittelmann's [AMPL-NLP benchmark](https://plato.asu.edu/ftp/ampl-nlp.html)
as [ExaModels](https://github.com/madsuite-org/ExaModels.jl), loadable on CPU or GPU.

Each instance is stored as an [ExaModelsIO](https://github.com/madsuite-org/ExaModels.jl/tree/main/ExaModelsIO)
file (`.h5`) in a lazy Julia artifact, downloaded the first time it is used. Two instances
are built by functions instead: `twod`, whose default size has about 126 million variables,
and `wm_cfy`.

```julia
using ExaMittelmann, CUDA

mittelmann_instances()                    # 121 instances
mittelmann_instances(; benchmark = true)  # the 46 of them scored in the benchmark table
mittelmann_file("clnlbeam")               # path of the .h5 file

m = mittelmann("clnlbeam")                         # CPU ExaModel
g = mittelmann("clnlbeam"; backend = CUDABackend()) # GPU ExaModel

t = twod(; n = 50)                         # twod at a chosen grid size
w = wm_cfy()                               # needs ExaModels with buffered subexpressions
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

The two function-built instances:

- `twod(; n = 500, m = n)` writes `twod.mod` directly in ExaModels. Single-variable
  constraints become variable bounds, so it has the fixed and unused variables that AMPL's
  presolve removes. At `n = 20`, MadNLP reaches the same optimal value as on AMPL's `.nl`
  (relative difference 1e-9 at tolerance 1e-10).
- `wm_cfy()` writes `WM_CFy.mod` directly, because the converted model's 1512 constraints
  that each contain `sqrt((1/N) sum_i y[i,k]^2)` did not compile. The two norms are buffered
  subexpressions, which ExaModels has only on the `compile-time-prototypes` branch so far; on other
  versions `wm_cfy()` raises an error. At a perturbed point, all 12850 constraint slacks agree
  with AMPL's evaluation to a relative 1.3e-15.

Until ExaModelsIO 0.1.1 is registered, ExaMittelmann takes ExaModelsIO from commit `e6af4e04`
of madsuite-org/ExaModels.jl, which reads data blocks with no columns (the constant terms of
these models); 0.1.0 does not.

## Packages in this repository

- `ExaMittelmann` (this directory): the instances.
- `MittelmannHelper` (`lib/MittelmannHelper`): the pipeline that builds the files,
  `ampl_to_nl`, `read_nl`, `to_core`, `check_core`, `save_model` (ExaModelsIO
  `write_model`), `check_file` (reads the file back and compares) and
  `bind_file_artifact!`. ExaMittelmann does not depend on it.
