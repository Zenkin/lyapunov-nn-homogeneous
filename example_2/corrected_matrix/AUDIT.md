# Audit of the corrected-matrix experiment

## Scope

The purpose of this experiment is to isolate the matrix correction. It reuses
the article-version nonlinear field, networks, pointwise loss, mean reduction,
optimizer, seed, grid convention, `kappa`, `epsilon`, and `W`-based switching
rule without modification.

The corrected Jacobian is

```text
A = [[0, 1],
     [1, 0]].
```

## Direct matrix-only result

Changing `A` while freezing the numerical `K` and `P` has no numerical effect.
The neural training and validation pipeline does not read `A`; it evaluates the
full nonlinear field directly. This is checked by a unit test that replaces
the matrix function with an exception during neural training.

The complete training and validation dictionaries are exactly equal to the
article-version records. The combined PNG files also have the same SHA-256
hash:

```text
ee688b85eb2593839048b1f9cc77e098a0b2c7678c419109239c440a8d9f7461
```

## Consistently recomputed local design

Keeping `K=(-2,-3)` and solving the Lyapunov equation for corrected `A` gives

```text
P = [[11/6, 1/2],
     [ 1/2, 1/3]],

(A+BK)^T P + P(A+BK) = -I.
```

This changes the `B_(kappa/2)` filter from 9,974 to 9,976 training points.
Everything else remains unchanged.

| Metric | Frozen old `P` | Corrected `P` |
| --- | ---: | ---: |
| Final mean pointwise loss | 0.0005348023 | 0.0005265847 |
| Final maximum pointwise loss | 0.1561239574 | 0.1565435080 |
| Minimum `W` on `X\\B_(kappa/2)` | 0.0074286798 | 0.0074374218 |
| Maximum `DW F` under learned control | 0.0559509756 | 0.0565700794 |
| Maximum `DW F+W` under learned control | 0.1505388518 | 0.1514240680 |
| Maximum local `DV_l F` on `B_kappa` | -0.0004930744 | -0.0009861488 |
| Maximum `DW F` under printed switching | 3.6885728651 | 5.2583135722 |
| Local-switch points outside `B_kappa` | 29 | 39 |

The corrected local quadratic design improves the sampled local derivative
margin, but the learned and switched inequalities still fail. Correcting the
matrix does not repair the outer neural optimization or the mismatch between
the switching set `{W<kappa}` and the local set `{V_l<=kappa}`.

## Literal switching-surface audit

Section IV-B does not require `W` and `V_l` to coincide. It states that they
belong to different controls and that a continuous Lyapunov function for the
switched system may differ from both. Therefore the first audit of the
printed construction must use its actual switching surface `W=kappa`, without
adding a gluing condition.

The unchanged corrected-local-design model was evaluated on successively
finer rectangular grids. Contours were extracted by marching squares and then
refined using the exact network gradient. The same saved model was used at
every resolution.

| Grid | Detected `W=kappa` contours | Components of `{W<kappa}` |
| ---: | ---: | ---: |
| 401 | 8 | 4 |
| 601 | 7 | 3 |
| 801 | 5 | 3 |
| 1201 | 5 | 3 |
| 1601 | 4 | 3 |
| 2001 | 3 | 3 |

The additional small contours disappear as the grid is refined and are not
treated as established geometry. At `2001 x 2001`, the three persistent
sampled components are one central component and two components touching the
boundaries `theta=-pi` and `theta=pi`.

On 1,519 refined points of the three detected level curves:

```text
max |W-kappa| after refinement                  = 3.4903655965479174e-08
max DW F(x,N(x))                                = 0.04593916716954357
fraction with DW F(x,N(x)) >= 0                 = 0.059907834101382486
max DW F(x,Kx)                                  = 6.84198862297304
fraction with DW F(x,Kx) >= 0                   = 0.4924292297564187
max V_l on W=kappa                              = 28.711074943157715
points on W=kappa outside B_kappa               = 1259
```

For the central level curve alone, `max V_l=0.42798076`, although
`kappa=0.05`. The local controller therefore becomes active outside the set
on which its local Lyapunov inequality was checked. Moreover,
`DW F(x,Kx)>=0` on approximately 46.9% of the sampled central contour. Under
the learned control, the same central contour has a positive maximum
derivative `0.04593917` and approximately 12.0% nonnegative points.

There are two distinct sign checks. Invariance of the inner sublevel set under
the local control would require `DW F(x,Kx)<=0` on its boundary; this fails on
approximately 49.2% of all sampled contour points. Invariance of the outer
superlevel set under the learned control would instead require
`DW F(x,N(x))>=0`; this fails wherever its derivative is negative, on
approximately 94.0% of the sampled contour. Conversely, the Lyapunov-descent
condition for the learned branch requires a negative derivative and fails on
the remaining approximately 6.0%. Thus the numerical data do not support the
article's parenthetical claim that both switching sets are invariant, and the
learned branch does not have strict descent on the entire detected boundary.

These are deterministic finite-grid counter-checks, not proofs over the
continuous state space. The contour-count convergence guards against
reporting the disappearing small grid artifacts as genuine components, but it
does not turn the numerical audit into a theorem.

The learned functions are also not periodic in `theta`: on the sampled seams
`theta=-pi` and `theta=pi`, the maximum mismatches are approximately `129.50`
for `W` and `5.90` for the learned control. Hence this network realization is
not a single-valued continuous controller on the physical cylinder unless the
angle is deliberately restricted to a local coordinate chart.

### Numerical self-check

At 41 deterministic contour points, the stored directional derivatives were
independently compared with the centered difference

```text
[W(x+hF(x))-W(x-hF(x))]/(2h),  h=1e-6.
```

The maximum absolute discrepancies were `1.88e-8` for the local field and
`1.81e-10` for the learned field. This checks the derivative implementation;
it does not address unsampled points or certify invariance.

## Inherited limitations deliberately retained

This controlled run retains two known bookkeeping/validation limitations of
`article_version` so that the matrix is the only changed mathematical input:

1. the `201 x 201` validation grid contains the `100 x 100` midpoint training
   grid as a subset and is not an independent sample;
2. the recorded final training loss is evaluated immediately before the last
   optimizer update, whereas validation is evaluated after it.

Neither run is a continuous-domain Lyapunov certificate.

## Exploratory controller-domain alignment

The following experiments are not requirements stated by the article. They
are retained as explicitly exploratory diagnostics performed after the
literal switching-surface audit.

For the printed switch to use the local controller only where its local check
is made, and the neural controller only where its loss is trained, the sets
would have to satisfy

```text
B_(kappa/2) subset {W < kappa} subset B_kappa.
```

On the `201 x 201` grid with the corrected `P`, both inclusions fail:

```text
neural-controller points inside B_(kappa/2) = 90
local-controller points outside B_kappa     = 39
```

The direct rule "use the local controller if `V_l<=kappa`, otherwise use the
neural controller" gives zero violations of both domain conditions by
construction. Its branchwise derivative record is

```text
max DV_l F_local on B_kappa          = -0.000986148769785377
max DW F_neural outside B_kappa      = +0.05657007937739486
fraction of active derivatives >= 0  = 0.0010891089108910892
```

Thus domain alignment repairs the switching-set mismatch but does not repair
the remaining outer derivative violations.

It also does not yet define a continuous composite candidate. On 2,048 exact
points of the ellipse `V_l=kappa=0.05`, the trained network gives

```text
min W = 0.007498021441280183
max W = 8.741305244960321
```

Consequently, the literal piecewise assignment `V=V_l` inside and `V=W`
outside would jump at the switching boundary. These are two separate issues:
controller-domain alignment is achieved by the level-set switch; Lyapunov
function gluing remains to be constructed and verified.

## Exploratory sampled boundary-matching experiment

The next experiment leaves the network architecture and the original
pointwise loss unchanged and adds

```text
lambda * mean(|W-kappa|) on 512 exact points of V_l=kappa.
```

`lambda` is not specified by the article. The values `0`, `0.1`, `1`, and `10`
were therefore run from the same initialization. Boundary generalization was
checked on 2,048 exact ellipse points shifted by half a validation-grid step;
none of those points is reused from boundary training.

| `lambda` | Original mean loss | Original max loss | Max boundary error | Max active derivative | Fraction active derivative `>=0` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0005262710 | 0.1565700794 | 8.6913052450 | 0.0565700794 | 0.0010891089 |
| 0.1 | 0.0014163826 | 0.1927332330 | 0.0006557106 | 0.0193720740 | 0.0008910891 |
| 1 | 0.0027902976 | 0.2458009527 | 0.0004471902 | 0.0604636741 | 0.0020792079 |
| 10 | 0.0070749558 | 0.4277932268 | 0.0003727059 | 0.2438041937 | 0.0106930693 |

Every nonzero tested weight gives zero controller-domain violations on the
`201 x 201` grid. Among these tested values, `lambda=0.1` has the smallest
sampled maximum active derivative. This is a comparison over four declared
values, not a proof that `0.1` is optimal. Its maximum derivative remains
positive, so none of the runs is a Lyapunov certificate.

The penalty also gives only approximate equality on sampled boundary points.
It does not establish exact continuity of a piecewise candidate on the entire
ellipse.
