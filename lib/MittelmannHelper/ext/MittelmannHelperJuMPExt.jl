module MittelmannHelperJuMPExt

import MittelmannHelper, ExaModels
import ExaModels: NLPModels
import JuMP
import JuMP: MOI

MittelmannHelper.read_nl(file::AbstractString) = JuMP.read_from_file(file; use_nlp_block = false)

_moi_ext() = Base.get_extension(ExaModels, :ExaModelsMOI)

# Flatten nested sums into one top-level `+`: `a + (b - (c + d))` -> `+(a, b, -c, -d)`,
# also distributing unary minus and constant factors (`2 * (a + b)` -> `+(2a, 2b)`).
# The converter splits a function only at its top-level `+`, so a nested sum would
# become a single expression tree with one type per term (bearing_200's 160800-term
# objective sum, dirichlet's `x - (sum of 450 terms)` rows), which does not compile
# in reasonable time.
_flatten(f) = f
function _flatten(f::MOI.ScalarNonlinearFunction)
    terms = Any[]
    _terms!(terms, f, 1.0)
    length(terms) == 1 && return only(terms)
    return MOI.ScalarNonlinearFunction(:+, terms)
end

_scale(t, c) = c == 1 ? t : t isa Real ? c * t : MOI.ScalarNonlinearFunction(:*, Any[c, t])

function _terms!(terms, f, c)
    if f isa MOI.ScalarNonlinearFunction
        h, a = f.head, f.args
        if h === :+
            foreach(x -> _terms!(terms, x, c), a)
            return terms
        elseif h === :- && length(a) == 1
            return _terms!(terms, a[1], -c)
        elseif h === :- && length(a) == 2
            _terms!(terms, a[1], c)
            return _terms!(terms, a[2], -c)
        elseif h === :* && length(a) == 2 && a[1] isa Real
            return _terms!(terms, a[2], c * a[1])
        elseif h === :* && length(a) == 2 && a[2] isa Real
            return _terms!(terms, a[1], c * a[2])
        end
    end
    push!(terms, _scale(f, c))
    return terms
end

function MittelmannHelper.to_core(model::JuMP.Model)
    JuMP.nonlinear_model(model) === nothing || throw(ArgumentError(
        "legacy @NL constraints are not supported; read .nl files with `read_nl`",
    ))
    # UniversalFallback keeps the starting point (VariablePrimalStart), which a plain
    # Utilities.Model drops during copy_to without an error
    moim = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
    index_map = MOI.copy_to(moim, JuMP.backend(model))
    F = MOI.get(moim, MOI.ObjectiveFunctionType())
    if F <: MOI.ScalarNonlinearFunction
        o = _flatten(MOI.get(moim, MOI.ObjectiveFunction{F}()))
        MOI.set(moim, MOI.ObjectiveFunction{F}(), o)
    end
    for (G, S) in MOI.get(moim, MOI.ListOfConstraintTypesPresent())
        G <: MOI.ScalarNonlinearFunction || continue
        for ci in MOI.get(moim, MOI.ListOfConstraintIndices{G, S}())
            MOI.set(moim, MOI.ConstraintFunction(), ci, _flatten(MOI.get(moim, MOI.ConstraintFunction(), ci)))
        end
    end
    # ExaModels' MOI optimizer numbers variables and constraint rows in the order they
    # are added, so its copy_to index map gives each one's position in the ExaModel.
    EMOI = _moi_ext()
    dest = EMOI.Optimizer{Float64}(nothing)
    index_map2 = MOI.copy_to(dest, moim)
    core = EMOI.to_exacore(dest, nothing)
    pos(k) = index_map2[index_map[k]].value
    var_to_idx = Dict(k => pos(k) for k in keys(index_map.var_map))
    con_to_idx = Dict(k => pos(k) for k in keys(index_map.con_map)
                      if !(k isa MOI.ConstraintIndex{MOI.VariableIndex}))
    return core, (var_to_idx, con_to_idx)
end

_bounds(s::MOI.EqualTo) = (s.value, s.value)
_bounds(s::MOI.GreaterThan) = (s.lower, Inf)
_bounds(s::MOI.LessThan) = (-Inf, s.upper)
_bounds(s::MOI.Interval) = (s.lower, s.upper)

# relative error; entries equal as values (including matching infinities) count as 0
_rel(a, b) = maximum((x == y ? 0.0 : abs(x - y) / (1 + abs(y)) for (x, y) in zip(a, b)); init = 0.0)

function MittelmannHelper.check_core(core, maps, model::JuMP.Model; x = nothing)
    em = ExaModels.ExaModel(core)
    var_to_idx, con_to_idx = maps
    vars = JuMP.index.(JuMP.all_variables(model))
    pv = [var_to_idx[v] for v in vars]                 # JuMP variable k -> ExaModel
    length(vars) == em.meta.nvar || error("variable counts differ")
    x0 = [something(JuMP.start_value(v), 0.0) for v in JuMP.all_variables(model)]
    xr = x === nothing ? x0 .+ 1e-3 .* (1 .+ abs.(x0)) .* sin.(1:length(x0)) : x
    xe = similar(xr)
    xe[pv] = xr
    val = Dict(zip(vars, xr))

    moim = JuMP.backend(model)
    # objective and gradient through MOI's reverse-mode AD
    nlp = MOI.Nonlinear.Model()
    F = MOI.get(moim, MOI.ObjectiveFunctionType())
    MOI.Nonlinear.set_objective(nlp, MOI.get(moim, MOI.ObjectiveFunction{F}()))
    ev = MOI.Nonlinear.Evaluator(nlp, MOI.Nonlinear.SparseReverseMode(), vars)
    MOI.initialize(ev, [:Grad])
    gr = zeros(length(xr))
    MOI.eval_objective_gradient(ev, gr, xr)
    fr = MOI.eval_objective(ev, xr)

    # each constraint function directly through MOI
    pc, cr, lr, ur = Int[], Float64[], Float64[], Float64[]
    for (F, S) in MOI.get(moim, MOI.ListOfConstraintTypesPresent())
        F <: MOI.VariableIndex && continue
        for ci in MOI.get(moim, MOI.ListOfConstraintIndices{F, S}())
            f = MOI.get(moim, MOI.ConstraintFunction(), ci)
            l, u = _bounds(MOI.get(moim, MOI.ConstraintSet(), ci))
            push!(pc, con_to_idx[ci]); push!(lr, l); push!(ur, u)
            push!(cr, MOI.Utilities.eval_variables(v -> val[v], moim, f))
        end
    end
    length(pc) == em.meta.ncon || error("constraint counts differ")
    ge = NLPModels.cons(em, xe)[pc]
    lcon, ucon = em.meta.lcon[pc], em.meta.ucon[pc]
    lvar = [JuMP.has_lower_bound(v) ? JuMP.lower_bound(v) :
            JuMP.is_fixed(v) ? JuMP.fix_value(v) : -Inf for v in JuMP.all_variables(model)]
    uvar = [JuMP.has_upper_bound(v) ? JuMP.upper_bound(v) :
            JuMP.is_fixed(v) ? JuMP.fix_value(v) : Inf for v in JuMP.all_variables(model)]
    return (
        obj = _rel([NLPModels.obj(em, xe)], [fr]),
        grad = _rel(NLPModels.grad(em, xe)[pv], gr),
        # rows compared as (g - l, u - g), which does not depend on where a constant is kept
        cons = _rel([ge .- lcon; ucon .- ge], [cr .- lr; ur .- cr]),
        bounds = _rel([em.meta.lvar[pv]; em.meta.uvar[pv]], [lvar; uvar]),
        x0 = _rel(em.meta.x0[pv], x0),
    )
end

end # module MittelmannHelperJuMPExt
