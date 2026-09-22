# Reproducibility summary

Generated directly from the committed JSON records. Epoch means one full-batch Adam update over the fixed points of the corresponding stage.

## Training protocol

| experiment | network | activation | parameters | optimizer | learning_rate | epochs | training_points | seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Example 1, homogeneous stage | 2-32-1 | tanh | 129 | Adam, full batch | 0.001 | 5000 | 2048 | 20260820 |
| Example 1, inner and transition stage | 2-32-2, anchored residual | tanh | 160 | Adam, full batch | 0.001 | 12000 | 9452 | 20260820 |
| Example 2, W and N trained jointly | W: 3-64-2; N: 3-32-1 | tanh | 544 | Adam, full batch | 0.002 | 6000 | 9952 | 20260820 |

## Numerical checks

| experiment | primary_validation | minimum_candidate | maximum_derivative | nonnegative_derivative_points | reported_level | trajectory_success |
| --- | --- | --- | --- | --- | --- | --- |
| Example 1 | 54140 mixed-grid points | 0.0127612237 | -0.0441434656 | 0 | V_inf = 2 kappa display boundary | 625/625 in X at t=40 |
| Example 2 | 401x401 plus 801x801 | 3.3250156e-05 | -6.15279486e-05 | 0 | V_NN <= 3.436 | 525/525 reached target |

## Execution environment

| experiment | matplotlib | numpy | platform | processor | python | pytorch | torch_interop_threads | torch_threads | scipy | wall_time_seconds | random_runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Example 1 | 3.10.8 | 2.3.5 | Linux-6.18.35-x86_64-with-glibc2.39 | x86_64 | 3.12.13 | 2.8.0+cpu | 1 | 1 | not used | 280.5502142250002 | 1 |
| Example 2 | 3.10.8 | 2.3.5 | Linux-6.18.35-x86_64-with-glibc2.39 | x86_64 | 3.12.13 | 2.8.0+cpu | 1 | 1 | 1.16.1 | 341.42156250100015 | 1 |

Both examples report one fixed-seed run. No across-seed mean or standard deviation is claimed. All grid results are finite-sample numerical evidence, not continuous-domain certificates.
