"""
    twod(; n = 500, m = n, backend = nothing, T = Float64, kwargs...)

The Mittelmann `twod` instance (plato.asu.edu/ftp/ampl-nlp-source/twod.mod), built
directly as an `ExaModel`: boundary control of the 2-D heat equation on an
`n × n` grid over `m` time steps. It is not shipped as an artifact because the
default `n = m = 500` has about 126 million variables. `kwargs` go to `ExaModel`.

Single-variable constraints of the source (`ic`, `cc`, `sc`) are variable bounds
here; AMPL's presolve does the same and also drops the fixed and unused variables,
so the `.nl` version has fewer variables but the same optimal value.
"""
function twod(; n = 500, m = n, backend = nothing, T = Float64, kwargs...)
    n1, m1 = n - 1, m - 1
    dx = 1 / n
    dy = dx
    dt = 1 / m
    h2 = dx^2
    a, ua, sa = 0.001, 2.0, 0.8
    yt(i, j) = 0.5 * i * dx * j * dy + 0.25
    # constant factors as products: in ExaModels, dividing a linear row by a constant
    # gives it Hessian entries, about 7e9 structural nonzeros at n = 500
    idt, ch, i2dx, i2dy = 1 / dt, 0.5 / h2, 1 / (2 * dx), 1 / (2 * dy)

    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # y[0, :, :] = 0 (ic), 0 <= y <= sa (sc)
    ylo = zeros(T, m + 1, n + 1, n + 1)
    yup = [k == 0 ? 0.0 : sa for k in 0:m, i in 0:n, j in 0:n]
    c, y = ExaModels.add_var(c, 0:m, 0:n, 0:n; lvar = ylo, uvar = yup)
    # 0 <= u <= ua (cc), starting at ua
    c, u = ExaModels.add_var(c, 1:m, 1:n1; start = ua, lvar = 0.0, uvar = ua)

    # terminal tracking: corner, edge and interior points weighted 1, 2, 4
    w(i) = i == 0 || i == n ? 1 : 2
    track = [(i, j, 0.125 * dx * dy * w(i) * w(j), yt(i, j)) for i in 0:n for j in 0:n]
    c, _ = ExaModels.add_obj(c, p[3] * (y[m, p[1], p[2]] - p[4])^2 for p in track)
    # control cost: steps 1..m-1 weighted 2, step m weighted 1
    ctrl = [(k, i, 0.25 * a * dt * dx * (k == m ? 1 : 2)) for k in 1:m for i in 1:n1]
    c, _ = ExaModels.add_obj(c, p[3] * u[p[1], p[2]]^2 for p in ctrl)

    # pde: Crank-Nicolson step of the heat equation
    pde = [(k, i, j) for k in 0:m1 for i in 1:n1 for j in 1:n1]
    c, _ = ExaModels.add_con(
        c,
        idt * (y[p[1]+1, p[2], p[3]] - y[p[1], p[2], p[3]]) -
        ch * (y[p[1], p[2], p[3]-1] - 4 * y[p[1], p[2], p[3]] + y[p[1], p[2], p[3]+1] +
               y[p[1], p[2]-1, p[3]] + y[p[1], p[2]+1, p[3]] + y[p[1]+1, p[2]-1, p[3]] +
               y[p[1]+1, p[2]+1, p[3]] + y[p[1]+1, p[2], p[3]-1] - 4 * y[p[1]+1, p[2], p[3]] +
               y[p[1]+1, p[2], p[3]+1]) for p in pde
    )

    # boundary conditions (second-order one-sided differences)
    side = [(k, i) for k in 1:m for i in 1:n1]
    c, _ = ExaModels.add_con(c,
        i2dy * (y[p[1], p[2], n-2] - 4 * y[p[1], p[2], n-1] + 3 * y[p[1], p[2], n]) +
        y[p[1], p[2], n] - u[p[1], p[2]] for p in side)                          # bc1
    c, _ = ExaModels.add_con(c,
        i2dy * (y[p[1], p[2], 2] - 4 * y[p[1], p[2], 1] + 3 * y[p[1], p[2], 0]) +
        y[p[1], p[2], 0] for p in side)                                          # bc2
    c, _ = ExaModels.add_con(c,
        i2dx * (y[p[1], 2, p[2]] - 4 * y[p[1], 1, p[2]] + 3 * y[p[1], 0, p[2]]) +
        y[p[1], 0, p[2]] for p in side)                                          # bc3
    c, _ = ExaModels.add_con(c,
        i2dx * (y[p[1], n-2, p[2]] - 4 * y[p[1], n-1, p[2]] + 3 * y[p[1], n, p[2]]) +
        y[p[1], n, p[2]] for p in side)                                          # bc4

    return ExaModels.ExaModel(c; kwargs...)
end
