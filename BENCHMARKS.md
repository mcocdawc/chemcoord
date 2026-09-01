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

## 4. Why there is only one Levenberg-Marquardt step

The back-transformation used to expose `lm_step` (`"full_step"`, `"line_search"`,
`"auto"`). The two variants existed because the truncated iterative solve handed both a
poor search direction, and each compensated differently. With the direct solve they are
the same algorithm in practice:

| workload | `full_step` | `line_search` |
|---|---|---|
| MIL53 + 101M, sigma 0.01 / 0.1 / 0.3, two seeds each | identical `\|dq\|` to every digit, identical wall clock | |
| cyclohexane `independent` | sum `\|W dq\|` = 0.354177 | 0.354177 |
| cyclohexane `from_start` | 0.361596 | 0.361596 |
| `default_args` interpolation | 0.975518 | 0.975518 |
| **101M, sigma = 0.5** | **raises `ConvergenceError`** | **converges** |

So the backtracking variant is never worse and is more robust at the hard end. It is
also the cheaper one to keep: `_linesearch` has to stay regardless, because
`_gauss_newton_opt` (`opt_alg="gauss"`) uses it, whereas keeping the full step would
mean carrying both the Armijo helper and a separate lambda-adaptation. `lm_step` was
removed along with `_full_step_cycle`, `_LMCycle`, `LMStep` and the auto-fallback,
together with the `sparse` switch and the `"RIC_sparse"`/`"RIC_dense"` spellings, for a
net -250 lines across the three modules.

Worth recording because it nearly went the other way: the full step was *expected* to
fail on interpolation, where the target `q` is unrealizable and no step can reduce the
residual at the minimum. It does not -- the three interpolation rows above are identical.
That reasoning would have picked the same variant for the wrong reason.

## 5. Comparison with published back-transformation algorithms

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

### `Gq = BBᵀ` against `Gx = BᵀB`, head to head

Farkas and Schlegel state that "the factorization of `Gq = BBᵀ` is less demanding than
the factorization of `Gx = BᵀB` in the screened Cholesky formalism". Both routes solve
the same problem, regularised by the same ridge (1e-8) so each matrix is factorisable:

```
Gx route:  (BᵀB + eps I) dx = Bᵀ dq             (3N x 3N)
Gq route:  (BBᵀ + eps I) y  = dq,  dx = Bᵀ y    (n_int x n_int)
```

Both converge to the minimum-norm least-squares solution; measured, they agree to
1e-7 and give identical residuals, so the timings are comparable.

| structure | atoms | matrix | dim | nnz | L+U nnz | fill | total time |
|---|---|---|---|---|---|---|---|
| MIL53_beta | 99 | `Gx` | 297 | 10 319 | 26 092 | 2.5x | **0.001 s** |
| | | `Gq` | 659 | 65 349 | 160 041 | 2.4x | 0.007 s |
| 101M | 1 413 | `Gx` | 4 239 | 113 823 | 282 825 | 2.5x | **0.007 s** |
| | | `Gq` | 5 763 | 178 901 | 341 552 | 1.9x | 0.011 s |
| 1A8I | 7 454 | `Gx` | 22 362 | 596 862 | 1 912 480 | 3.2x | **0.045 s** |
| | | `Gq` | 30 037 | 903 633 | 1 969 622 | 2.2x | 0.054 s |
| 1B0P | 19 411 | `Gx` | 58 233 | 1 557 045 | 3 804 541 | 2.4x | **0.099 s** |
| | | `Gq` | 78 424 | 2 343 222 | 4 439 439 | 1.9x | 0.130 s |
| 1A2V | 33 726 | `Gx` | 101 178 | 2 738 718 | 8 770 437 | 3.2x | **0.204 s** |
| | | `Gq` | 139 038 | 4 316 954 | 9 902 118 | 2.3x | 0.262 s |

`Gx` is faster at every size. The margin narrows sharply from small to medium systems
(7.0x, 1.6x, 1.2x) and then settles at about 1.3x; `Gq` never overtakes.

Their claim is not baseless, though, and the fill column shows why: `Gq` factorises
*better per nonzero* at every size (1.9-2.4x against 2.4-3.2x). Despite being 34-38%
larger in dimension and carrying 1.5-1.6x the nonzeros, its `L+U` is only 3-13% bigger.
What sinks it here is that it starts from more nonzeros, not that it factorises worse.

**Caveat.** This compares the two matrices under one general-purpose sparse LU with
COLAMD ordering. It is *not* their algorithm. Their screening drops the rows with
near-zero pivots -- the redundant ones -- which would shrink `Gq` from `n_int` to about
`3N - 6`, i.e. to `Gx`'s dimension while keeping `Gq`'s better fill behaviour. On 1A2V
that would mean dropping ~37 900 of 139 038 rows. They also stress that "proper
reordering is essential for the efficiency of the factorization" and use a
divide-and-conquer scheme after Nemeth et al. Either could flip this verdict, and
neither is implemented here.

### Does the internal-coordinate weighting matter?

`W` (bond 1.0, angle 0.1, dihedral 0.05, bending 0.01) only changes the answer when the
target `q` is **not realizable** by any structure. If some `x` has `q(x) = q`, the
residual is zero and every positive-definite `W` has that `x` as a minimiser; if no such
`x` exists, `W` chooses which compromise you land on.

Cyclohexane, same seed, solved with the default weights and with uniform weights:

| target | structures differ by |
|---|---|
| realizable (`q = q(m1)`, a real structure) | 3.1e-15 A -- identical |
| unrealizable (interpolation midpoint) | **3.1e-02 A** |

For the unrealizable target, the weighted residual `‖W dq‖` of each converged structure,
scored under both weightings:

| | under default `W` | under uniform `W` |
|---|---|---|
| solved with default `W` | **2.794573e-02** | 3.055156e-01 |
| solved with uniform `W` | 7.294749e-02 | **2.786186e-01** |

Each is the better structure under its own weighting, by a factor of 2.6 in the default
case. So the weights are a real modelling choice for interpolation -- "distort dihedrals
rather than bonds" -- and a no-op for an ordinary back-transformation.

This is also why Farkas and Schlegel have no weighting at all (the word does not appear
in the paper): in geometry optimization the target is a quasi-Newton step from a real
structure, so the problem is effectively consistent and `W` would not change the answer.
chemcoord's interpolation poses a genuinely inconsistent problem, where it does.

Note also that they factorise the *internal-space* matrix by preference -- "the
factorization of `Gq = BBᵀ` is less demanding than the factorization of `Gx = BᵀB` in the
screened Cholesky formalism" -- which is the opposite of the choice made here. Their
`B` is unweighted and their screening drops the redundant rows; with the LM damping the
Cartesian-side matrix is positive definite and needs neither. The two routes have not
been compared head to head on the same code.

### Prior art

The approach used here is **not new**. Farkas and Schlegel (2003) already solve the
coordinate transformation with a screened sparse Cholesky decomposition, and discuss
exactly this matrix:

> "Paizs et al. pointed out that any set of redundant internal coordinates could be
> constructed from their complete, but non-redundant, subsets as linear combinations.
> They also concluded that this also applies to matrices `BᵀB` and `BBᵀ`, and their
> rows. The full Cholesky factorization of positive semi-definite matrices results in
> zero diagonal values. Because of consequent divisions by zero, the full Cholesky
> factorization ... can only be applied to positive definite matrices. The zero (or in
> practice very small) diagonal values, however, indicate rows that can be pro[duced as
> linear combinations of others]."

They handle the semi-definiteness by *screening*: a near-zero pivot marks a redundant
row, which is dropped. The only difference here is that the Levenberg-Marquardt damping
`lambda D^2` already makes the matrix positive definite, so no screening is needed and a
plain `splu` suffices. The measurements above are therefore a confirmation on this
codebase, not a new method.

### References

- LSMR, the solver replaced: D. C.-L. Fong and M. Saunders, *LSMR: An iterative
  algorithm for sparse least-squares problems*, SIAM J. Sci. Comput. **33**, 2950 (2011),
  arXiv:1006.0758. It is built on Golub-Kahan bidiagonalization and is analytically
  equivalent to MINRES applied to the normal equations `AᵀA x = Aᵀb` -- so the old code
  was already solving these normal equations, just iteratively. Its predecessor is LSQR:
  C. C. Paige and M. A. Saunders, ACM Trans. Math. Softw. **8**, 43 (1982).

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
- O. Farkas and H. B. Schlegel, *Geometry optimization methods for modeling large
  molecules*, J. Mol. Struct. THEOCHEM **666-667**, 31 (2003). Screened sparse Cholesky
  for the coordinate transformation -- the prior art for what this branch does.
- B. Paizs, J. Baker, S. Suhai, P. Pulay, *Geometry optimization of large biomolecules in
  redundant internal coordinates*, J. Chem. Phys. **113**, 6566 (2000).
- K. Nemeth, O. Coulaud, G. Monard, J. G. Angyan, *Linear scaling algorithm for the
  coordinate transformation problem of molecular geometry optimization*, J. Chem. Phys.
  **113**, 5598 (2000); *An efficient method for the coordinate transformation problem of
  massively three-dimensional networks*, J. Chem. Phys. **114**, 9747 (2001).

## 6. Stability against increasingly perturbed seeds

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

## 7. The iterative solve, and why it was replaced

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

## 8. Levers that did not work

Same case, `line_search`:

| lever | result |
|---|---|
| `atol`/`btol` 1e-8 -> 1e-6 | identical (1268 iters, 49 s). LSMR never approaches tolerance, so the target is irrelevant. |
| Jacobi (column-norm) scaling | worse: 1627 iters, 63 s, `max\|dx\|` 2.01e-03 |
| LM damping floor 1e-14 -> 1e-6 / 1e-4 / 1e-2 | no change in LSMR iterations; over-damping degrades the line search (`\|dq\|` after 60 iters: 1.90e-04 -> 4.45e-03 at floor 1e-2) |

The ill-conditioning comes from the redundancy of the coordinate set, not from column
scaling or from the damping schedule.

## 9. `line_search` vs `full_step` (with the iterative solve)

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
