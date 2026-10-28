"""
    wm_cfy(; backend = nothing, T = Float64, kwargs...)

The Mittelmann `WM_CFy` instance (plato.asu.edu/ftp/ampl-nlp-source/WM_CFy.mod),
built directly as an `ExaModel`. `kwargs` go to `ExaModel`.

The norms `sqrt((1/N) sum_i y[i,k]^2)` that appear in 1512 constraint rows are
buffered subexpressions, summed in two levels (21 blocks of 18 terms), so this
needs an ExaModels with `add_expr(...; buffered = true)`. Single-variable
constraints of the source (`bound51`..`bound62`) are variable bounds, as in AMPL's
presolve. At a perturbed starting point, every one of the other 12850 constraint
slacks agrees with AMPL's own evaluation to a relative 1.3e-15.
"""
function wm_cfy(; backend = nothing, T = Float64, kwargs...)
    N, M, Nu = 378, 16, 189
    mx, my, bd = 0.01, 0.008, 0.0085
    pi_ = 4 * atan(1)
    pn(i) = i * 2 * pi_ / N
    isdefined(ExaModels, :BufferedExpression) || error(
        "wm_cfy needs buffered subexpressions (`add_expr(...; buffered = true)`), which " *
        "ExaModels $(pkgversion(ExaModels)) does not have",
    )
    d = TOML.parsefile(joinpath(@__DIR__, "WM_CFy.toml"))
    rows(v) = permutedims(reduce(hcat, v))
    a1, a2, alpha0 = Float64.(d["a1"]), Float64.(d["a2"]), rows(d["alpha0"])
    A, B, C = rows(d["A"]), rows(d["B"]), rows(d["C"])
    a = hcat(a1, a2)

    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    c, alpha = ExaModels.add_var(c, 1:Nu, 1:2; start = alpha0)
    c, x = ExaModels.add_var(c, 0:N-1, 1:2; start = 1.0)
    c, z = ExaModels.add_var(c, 0:N, 1:M; start = 1.0)
    ylo = [i >= N ? -bd : -Inf for i in 0:2N-1, j in 1:2]
    c, y = ExaModels.add_var(c, 0:2N-1, 1:2; start = 1.0, lvar = ylo, uvar = -ylo)
    c, t = ExaModels.add_var(c, 1)
    c, _ = ExaModels.add_obj(c, t[1] for _ in 1:1)

    # r[k] = sqrt((1/N) * sum_{i=N}^{2N-1} y[i,k]^2), as two levels of buffered sums
    # (21 blocks of 18 terms) so no expression node has more than 21 children
    nb, bs = 21, 18                                   # nb * bs == N
    c, q = ExaModels.add_expr(c, ExaModels.exa_sum(l -> y[N + bs * (p[1] - 1) + l, p[2]]^2, Val(0:bs-1))
                              for p in [(b, k) for b in 1:nb, k in 1:2]; buffered = true)
    c, r = ExaModels.add_expr(c, sqrt((1 / N) * ExaModels.exa_sum(b -> q[b, k], Val(1:nb)))
                              for k in 1:2; buffered = true)
    rows = [(j, k) for k in 1:2 for j in N:2N-1]
    # bound31/41: -t r_k <= y[j,k];  bound32/42: y[j,k] <= t r_k
    c, _ = ExaModels.add_con(c, y[p[1], p[2]] + t[1] * r[p[2]] for p in rows; lcon = 0.0, ucon = Inf)
    c, _ = ExaModels.add_con(c, t[1] * r[p[2]] - y[p[1], p[2]] for p in rows; lcon = 0.0, ucon = Inf)

    # inp: x[i,j] = sum_u a_j[u] cos(pn(i) u + alpha[u,j])
    c, inp = ExaModels.add_con(c, x[p[1], p[2]] for p in [(i, j) for i in 0:N-1 for j in 1:2])
    terms = [(2i + j, a[u, j], pn(i) * u, u, j) for i in 0:N-1 for j in 1:2 for u in 1:Nu]
    c, _ = ExaModels.add_con!(c, inp, p[1] => -p[2] * cos(p[3] + alpha[p[4], p[5]]) for p in terms)

    # out1: z[i,j] = sum_l A[j,l] z[i-1,l] + sum_l B[j,l] x[i-1,l]
    out1 = [(i, j) for i in 1:N for j in 1:M]
    c, o1 = ExaModels.add_con(c, z[p[1], p[2]] for p in out1)
    c, _ = ExaModels.add_con!(c, o1, p[1] => -p[2] * z[p[3], p[4]]
        for p in [(M * (i - 1) + j, A[j, l], i - 1, l) for i in 1:N for j in 1:M for l in 1:M])
    c, _ = ExaModels.add_con!(c, o1, p[1] => -p[2] * x[p[3], p[4]]
        for p in [(M * (i - 1) + j, B[j, l], i - 1, l) for i in 1:N for j in 1:M for l in 1:2])

    # out2: y[i,j] = sum_l C[j,l] z[i,l];  out3: y[i,j] = sum_l C[j,l] z[i-N-1,l]
    for (irange, shift) in ((1:N, 0), (N+1:2N-1, N + 1))
        rws = [(i, j) for i in irange for j in 1:2]
        c, o = ExaModels.add_con(c, y[p[1], p[2]] for p in rws)
        c, _ = ExaModels.add_con!(c, o, p[1] => -p[2] * z[p[3], p[4]]
            for p in [(2 * (i - first(irange)) + j, C[j, l], i - shift, l)
                      for i in irange for j in 1:2 for l in 1:M])
    end

    # move1..move4, movey, movey1: one-sided rows, in the source's order
    xs = [(i, j) for i in 0:N-2 for j in 1:2]
    ys = [(i, j) for i in N-1:2N-2 for j in 1:2]
    c, _ = ExaModels.add_con(c, x[p[1]+1, p[2]] - x[p[1], p[2]] for p in xs; lcon = -mx, ucon = Inf)  # move1
    c, _ = ExaModels.add_con(c, x[0, j] - x[N-1, j] for j in 1:2; lcon = -mx, ucon = Inf)            # move2
    c, _ = ExaModels.add_con(c, y[p[1]+1, p[2]] - y[p[1], p[2]] for p in ys; lcon = -my, ucon = Inf)  # movey
    c, _ = ExaModels.add_con(c, x[p[1]+1, p[2]] - x[p[1], p[2]] for p in xs; lcon = -Inf, ucon = mx)  # move3
    c, _ = ExaModels.add_con(c, x[0, j] - x[N-1, j] for j in 1:2; lcon = -Inf, ucon = mx)            # move4
    c, _ = ExaModels.add_con(c, y[p[1]+1, p[2]] - y[p[1], p[2]] for p in ys; lcon = -Inf, ucon = my)  # movey1
    return ExaModels.ExaModel(c; kwargs...)
end
