# Back-transformation benchmarks

Measurements behind the choices in
`src/chemcoord/_redundant_internal_coordinates/_backtransformation.py`, in particular
`_LSTSQ_MAX_ITER`. Absolute times are machine-dependent; the ratios are the point.

**Setup.** Apple M3 Pro, python 3.12.13, numpy 2.4.4 (Accelerate BLAS), scipy 1.17.1.

**Benchmark case** unless stated otherwise: `tests/structures/101M.xyz` (1413 atoms,
5763 primitive internal coordinates), 1% of atoms displaced by up to 0.1 A:

```python
rng = np.random.default_rng(0)
n, k = len(m1), round(0.01 * len(m1))
x = np.zeros((n, 3))
x[rng.choice(n, size=k, replace=False)] = np.clip(rng.normal(0, 0.1, (k, 3)), -0.1, 0.1)
m2 = m1 + x
q1.get_cartesian(start_guess=m2, opt_alg="LM", lm_step=..., max_iter=2000)
```

`|dq|` is the unweighted residual against the target internals; `max|dx|` is the largest
per-atom coordinate deviation from the true structure after alignment.

## 1. Wilson B correctness

Finite differences, `(q(x + h*d) - q(x)) / h` against `B @ d` for a random unit `d` and
`h = 1e-7`, as relative error over each coordinate block:

| block | before the dihedral fix | after |
|---|---|---|
| bond (`default_args_start` / `cyclohexane_chair`) | 2.7e-08 / 4.5e-08 | unchanged |
| angle | 1.9e-08 / 3.3e-08 | unchanged |
| **dihedral** | **9.2e-01 / 3.1e-01** | **2.0e-07 / 1.8e-08** |

The dihedral rows were wrong for the two central atoms only, by equal and opposite
amounts, so the rows still summed to zero and the defect survived the obvious
translational-invariance check.

## 2. What the dihedral fix bought

At `_LSTSQ_MAX_ITER = 200`, `max_iter=1000`:

| case | before | after |
|---|---|---|
| MIL53_beta, sigma=0.2 | `ConvergenceError` | `max\|dx\|` = 2.9e-07 |
| 101M, sigma=0.01, all atoms | `\|dq\|` = 1.37e-02 | **4.50e-05** |
| lambda-cap give-ups in `default_args` | 6 | **0** |
| `lm_step="auto"` fallbacks in `default_args` | 6 | **0** |
| `default_args` residual sum | 0.999780 | **0.975518** |

## 3. Direct sparse factorisation (current solver)

The linear solve is a direct sparse LU of the damped normal equations,
`(AᵀA) x = Aᵀb`, where `A = [W B; sqrt(lambda) D]` is the augmented
Levenberg-Marquardt system. `AᵀA` is the much smaller `(3 n_atoms, 3 n_atoms)` matrix
and, being a molecular connectivity graph squared, barely fills in.

One representative 101M solve (augmented system 10002 x 4239, 58885 nonzeros;
normal equations 4239 x 4239, 113823 nonzeros, density 6.3e-03):

| solver | time | relative error vs the exact solution |
|---|---|---|
| `lsmr`, cap 2000 | 0.219 s | 7.9e-03 |
| **`splu` on the normal equations** | **0.007 s** (0.006 factorise + 0.001 solve) | **8.1e-08** |
| CG + Jacobi on the normal equations | 0.405 s | did not converge in 5000 iterations |

L+U has 248802 nonzeros, a fill-in of 2.2x.

End to end, back-transforming a perturbed 101M structure, `max_iter=2000`:

| perturbation | `lm_step` | outer iters | `\|dq\|` | `max\|dx\|` | time |
|---|---|---|---|---|---|
| 1% of atoms, 0.1 A | all three | < 100 | 1.67e-12 | 2.25e-10 | 0.2 s |
| all atoms, 0.01 A | all three | < 100 | 1.81e-13 | 6.40e-14 | 0.2 s |
| all atoms, 0.1 A | all three | < 100 | 1.88e-13 | 5.99e-13 | 0.2 s |

For comparison, the same "all atoms, 0.01 A" case took 2.9e-05 in 55 s with `lsmr`, and
2.3e-03 in 80 s before the dihedral fix. With an exact solve the three `lm_step`
strategies become indistinguishable -- the step control only ever mattered because it
was compensating for a truncated direction.

Normal equations square the condition number, which is the usual reason to avoid them.
It is tolerable here because the LM damping regularises the system, and it is measured
rather than assumed: the error above is five orders of magnitude below the iterative
solve it replaced. `lsmr` remains as a fallback for a singular factorisation; across the
test suite it is never reached (0 fallbacks in 49 solves).

`_full_step_cycle` also needed the classic LM damping decay after an accepted step.
Without it, one overshoot raised the damping and nothing brought it back; with an exact
solve that stalled `full_step` at 2000 outer iterations without converging. With the
decay it matches the other two at 0.2 s.

## 4. The iterative solve, and why it was replaced

All runs converged to the same outer-loop tolerance.

| config | outer iters | `\|dq\|` | `max\|dx\|` | time |
|---|---|---|---|---|
| cap 200, `full_step` | 1503 | 2.85e-05 | 1.28e-03 | 57 s |
| cap 200, `line_search` | 1268 | 2.91e-05 | 1.29e-03 | 48 s |
| cap 2000, `line_search` | 582 | 1.15e-06 | 1.54e-04 | 135 s |
| cap 2000, `auto` | 495 | 1.19e-06 | 1.49e-04 | 127 s |
| cap 2000, `full_step` | — | — | — | > 290 s, not run to completion |
| cap 5000, `line_search` | — | — | — | > 280 s, not run to completion |

Residual reached after a fixed budget of 60 outer iterations, from a starting
`|dq|` of 9.43e-01:

| cap | `\|dq\|` after 60 | time | residual drop per second |
|---|---|---|---|
| 200 | 1.90e-04 | 2.3 s | **4.07e-01** |
| 2000 | 1.62e-05 | 14.0 s | 6.75e-02 |
| 5000 | 2.08e-06 | 33.1 s | 2.85e-02 |

Per second of wall clock the truncated solve is the more productive one, by 14x at cap
5000. There is no crossover: a more accurate solve is always slower to a given residual.

**But the cap is an accuracy floor, not just a speed knob.** Truncating the solve
shortens every step, so the outer loop's `allclose(new, previous)` test trips while the
structure is still short of the minimum. At cap 200 the converged answer is 1.3e-03 A
off and no iteration budget recovers it; at cap 2000 it is 1.5e-04 A. That is why the
default is 2000 despite costing ~2.8x the wall clock.

For scale: one representative solve of this system (10002 x 4239 after augmentation,
58885 nonzeros) needs **3010** LSMR iterations to reach its stopping tolerance. Every
solve in this benchmark terminates on `maxiter`, never on tolerance. Over those extra
iterations the residual barely moves (`|Ax-b|` 2.401e-03 -> 2.271e-03); it is the
solution vector that changes (`|x|` 0.418 -> 0.455).

## 5. Levers that did not work

Same case, `line_search`:

| lever | result |
|---|---|
| `atol`/`btol` 1e-8 -> 1e-6 | identical (1268 iters, 49 s). LSMR never approaches tolerance, so the target is irrelevant. |
| Jacobi (column-norm) scaling | worse: 1627 iters, 63 s, `max\|dx\|` 2.01e-03 |
| LM damping floor 1e-14 -> 1e-6 / 1e-4 / 1e-2 | no change in LSMR iterations; over-damping degrades the line search (`\|dq\|` after 60 iters: 1.90e-04 -> 4.45e-03 at floor 1e-2) |

The ill-conditioning comes from the redundancy of the coordinate set, not from column
scaling or from the damping schedule.

## 6. `line_search` vs `full_step` (with the iterative solve)

`line_search` is faster, but only by ~16% (1268 vs 1503 outer iterations). The step
control cannot matter more than that while both strategies are handed the same
15x-truncated direction. It pulls further ahead once the solve is accurate: at cap 2000
it reaches 1.62e-05 after 60 iterations where `full_step` reaches 1.10e-04.

## Reproducing

The probes used here are not part of the test suite. The finite-difference check in
section 1 is the one worth keeping to hand: build `B = m.get_Wilson_B(idx)`, take a
random unit displacement `d`, and compare `B @ d` against
`(m2.get_ric(idx) - m.get_ric(idx)).minimize_dihedral().delta_q / h`. Bonds and angles
should agree to ~1e-08; so should dihedrals.
