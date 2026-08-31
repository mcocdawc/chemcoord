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

## 4. Comparison with published back-transformation algorithms

**Caveat on faithfulness.** These are the *linear-solve kernels* of the published
methods, not the published algorithms. The divide-and-conquer fragmentation of Billeter,
Turner & Thiel (HDLC, 2000) and the specific scheme of Farkas & Schlegel (1998) are whole
algorithms that were not reimplemented. In particular the "Nemeth-style" row below is a
naive application of preconditioned CG with an incomplete factorisation to the singular
internal-space matrix; the published method has machinery this lacks, so its failure here
should be read as "the internal-space route needs that machinery", not as a verdict on
their work.

All kernels solve the same damped system `A = [W B; sqrt(lambda) D]`, `b = [W dq; 0]`
with `lambda = 1e-5`, so the solution is unique and the relative error is well defined.
Reference is a dense direct solve of the normal equations.

MIL53_beta (99 atoms, 3N = 297, 659 internals):

| kernel | time | rel. error | residual |
|---|---|---|---|
| Pulay/Fogarasi: dense `pinv(A Aᵀ)` | 0.094 s | 3.9e-12 | 6.060985e-04 |
| dense LAPACK `lstsq` | 0.012 s | 5.6e-12 | 6.060985e-04 |
| LSMR, cap 2000 | 0.044 s | 2.9e-04 | 6.060985e-04 |
| LSMR to convergence | 0.056 s | 5.2e-10 | 6.060985e-04 |
| Nemeth-style: PCG + ILU on `A Aᵀ` | 0.258 s | did not converge | — |
| PCG + ILU on the normal equations | 0.001 s | 3.5e-08 | 6.060985e-04 |
| **sparse LU on the normal equations** | **0.001 s** | **4.5e-12** | 6.060985e-04 |

101M (1413 atoms, 3N = 4239, 5763 internals):

| kernel | time | rel. error | residual |
|---|---|---|---|
| dense `pinv(A Aᵀ)` / dense `lstsq` | — | skipped: `A Aᵀ` dense is 0.8 GB | |
| LSMR, cap 2000 | 0.215 s | 7.9e-03 | 2.270973e-03 |
| LSMR to convergence | 0.681 s | 8.1e-08 | 2.270959e-03 |
| Nemeth-style: PCG + ILU on `A Aᵀ` | 1.099 s | did not converge | — |
| PCG + ILU on the normal equations | 0.384 s | 2.6e-04 | 2.270959e-03 |
| **sparse LU on the normal equations** | **0.005 s** | **8.5e-12** | 2.270959e-03 |

The structural reason the internal-space route is hard: `A Aᵀ` is `n_int x n_int` and its
rank deficiency is the *redundancy* of the coordinate set. On cyclohexane, `B^T W^2 B` is
54 x 54 with exactly 6 zero eigenvalues -- the rigid-body motions, confirmed by a
1.000000 overlap with the translation/rotation span -- while `G = B W^2 B^T` is 108 x 108
with **60** zero eigenvalues. The Cartesian-side matrix never sees the redundancy,
because multiplying by `Bᵀ` projects onto the row space of B, and the LM damping lifts
the remaining 6. That is what makes a plain direct factorisation viable with no
pseudo-inverse and no preconditioner.

### Scaling

Direct factorisation of the normal equations, 1% of atoms perturbed:

| structure | atoms | 3N | `N` nnz | fill-in | build | factorise | solve | LSMR @2000 |
|---|---|---|---|---|---|---|---|---|
| MIL53_beta | 99 | 297 | 10 319 | 2.6x | 0.000 s | 0.001 s | 0.000 s | 0.043 s |
| 101M | 1 413 | 4 239 | 113 823 | 2.2x | 0.001 s | 0.005 s | 0.000 s | 0.215 s |
| 1A8I | 7 454 | 22 362 | 596 862 | 3.0x | 0.006 s | 0.036 s | 0.001 s | 0.825 s |
| 1B0P | 19 411 | 58 233 | 1 557 045 | 2.3x | 0.014 s | 0.077 s | 0.002 s | 1.961 s |
| 1A2V | 33 726 | 101 178 | 2 738 718 | 2.9x | 0.029 s | 0.163 s | 0.005 s | 3.507 s |

341x the atoms costs 163x the factorisation time -- linear scaling in practice, matching
what the published linear-scaling methods achieve, but with a direct solver rather than a
preconditioned iterative one. Fill-in stays at 2-3x at every size: `BᵀB` couples two
atoms iff they share an internal coordinate, so its pattern is the molecular connectivity
graph squared, which stays sparse and orders well.

### References

- P. Pulay and G. Fogarasi, *Geometry optimization in redundant internal coordinates*,
  J. Chem. Phys. **96**, 2856 (1992).
- J. Baker, A. Kessi, B. Delley, *The generation and use of delocalized internal
  coordinates in geometry optimization*, J. Chem. Phys. **105**, 192 (1996).
- O. Farkas and H. B. Schlegel, *Methods for geometry optimization of large molecules.
  I. An O(N^2) algorithm for solving systems of linear equations for the transformation
  of coordinates and forces*, J. Chem. Phys. **109**, 7100 (1998).
- S. R. Billeter, A. J. Turner, W. Thiel, *Linear scaling geometry optimisation and
  transition state search in hybrid delocalised internal coordinates*, Phys. Chem. Chem.
  Phys. **2**, 2177 (2000).
- K. Nemeth, O. Coulaud, G. Monard, J. G. Angyan, *Linear scaling algorithm for the
  coordinate transformation problem of molecular geometry optimization*, J. Chem. Phys.
  **113**, 5598 (2000); *An efficient method for the coordinate transformation problem of
  massively three-dimensional networks*, J. Chem. Phys. **114**, 9747 (2001).

## 5. Stability against increasingly perturbed seeds

Every atom displaced by a clipped Gaussian of width sigma, then back-transformed from
that seed towards the *original* structure's internals. `max|dx|` is measured against the
original, so it answers "did we get the structure back", not merely "did it converge".
Three seeds per point, `max_iter=2000`.

MIL53_beta (99 atoms), direct sparse LU:

| sigma [A] | start `\|dq\|` | converged | final `\|dq\|` | `max\|dx\|` vs original | time |
|---|---|---|---|---|---|
| 0.001 | 3.4e-02 | 3/3 | 1.6e-08 .. 2.2e-08 | 8.3e-09 .. 2.1e-08 | 0.1 s |
| 0.01 | 3.4e-01 | 3/3 | 1.9e-08 .. 3.2e-08 | 7.5e-09 .. 1.7e-08 | 0.1 s |
| 0.03 | 1.0e+00 | 3/3 | 2.4e-08 .. 2.8e-08 | 6.2e-09 .. 3.7e-08 | 0.1 s |
| 0.1 | 3.4e+00 | 3/3 | 2.6e-08 .. 3.0e-08 | 3.5e-09 .. 2.0e-08 | 0.1 s |
| 0.3 | 1.2e+01 | 3/3 | 1.4e-08 .. 3.6e-08 | 1.5e-08 .. 2.5e-08 | 0.1 s |
| 0.5 | 2.1e+01 | 3/3 | 2.5e-08 .. **4.2e+00** | 2.0e-08 .. **1.5e+00** | 0.2 s |
| 1.0 | 3.2e+01 | 3/3 | 4.2e+00 .. 6.8e+00 | 1.5e+00 .. 3.1e+00 | 0.3 s |

101M (1413 atoms), direct sparse LU:

| sigma [A] | start `\|dq\|` | converged | final `\|dq\|` | `max\|dx\|` vs original | iters | time |
|---|---|---|---|---|---|---|
| 0.001 | 9.8e-02 | 3/3 | 2.4e-13 .. 6.0e-13 | 2.2e-11 .. 8.9e-11 | < 100 | 0.2 s |
| 0.01 | 9.8e-01 | 3/3 | 1.9e-13 .. 1.5e-12 | 1.0e-13 .. 2.2e-10 | < 100 | 0.2 s |
| 0.03 | 2.9e+00 | 3/3 | 1.7e-13 .. 1.8e-13 | 9.2e-14 .. 4.0e-13 | < 100 | 0.2 s |
| 0.1 | 9.7e+00 | 3/3 | 1.8e-13 .. 1.9e-13 | 2.3e-13 .. 1.1e-12 | < 100 | 0.2 s |
| 0.3 | 3.1e+01 | 3/3 | 1.9e-13 .. 1.9e-13 | 2.3e-12 .. 3.9e-12 | < 100 | 0.2 s |
| 0.5 | 5.5e+01 | 3/3 | **7.9e+00 .. 8.7e+00** | **2.7e+01 .. 4.4e+01** | 0 - 552 | 11.8 s |
| 1.0 | 9.1e+01 | 2/3 | 2.2e+01 .. 2.4e+01 | 4.3e+01 .. 4.7e+01 | 473 - 709 | 20.8 s | 

Three conclusions:

**The basin is a property of the problem, not of the solver.** Running the identical
MIL53 sweep with the old LSMR solver reproduces the recovery/failure pattern at *every*
sigma, with the same final values -- perfect recovery to 0.3, the edge at 0.5 (two seeds
of three), outside it at 1.0 -- and differs only in taking 4-8x as long. What sets the
radius is the nonlinearity of the coordinate map, not the accuracy of the linear solve.

**Inside the basin the solver decides the accuracy, and the margin is large.** On 101M at
sigma = 0.01: LSMR reaches `max|dx|` = 2.8e-04 A in 580 iterations and 146 s; the direct
solve reaches 1e-13 A in under 100 iterations and 0.2 s. Roughly 730x faster and nine
orders of magnitude closer. Recovery stays at machine precision out to sigma = 0.3 A,
where the starting residual is already 30.6.

**Degradation is graceful.** Past the basin the solve still converges; it converges to a
*different* structure, which the residual reports honestly (7.9e+00 rather than 1e-13).
The one hard failure, at sigma = 1.0 on 101M, is an `UndefinedDihedral` raised by the
coordinate definition when the perturbation makes three atoms collinear -- a property of
the internal coordinate set, not of the back-transformation.

## 6. The iterative solve, and why it was replaced

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

## 7. Levers that did not work

Same case, `line_search`:

| lever | result |
|---|---|
| `atol`/`btol` 1e-8 -> 1e-6 | identical (1268 iters, 49 s). LSMR never approaches tolerance, so the target is irrelevant. |
| Jacobi (column-norm) scaling | worse: 1627 iters, 63 s, `max\|dx\|` 2.01e-03 |
| LM damping floor 1e-14 -> 1e-6 / 1e-4 / 1e-2 | no change in LSMR iterations; over-damping degrades the line search (`\|dq\|` after 60 iters: 1.90e-04 -> 4.45e-03 at floor 1e-2) |

The ill-conditioning comes from the redundancy of the coordinate set, not from column
scaling or from the damping schedule.

## 8. `line_search` vs `full_step` (with the iterative solve)

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
