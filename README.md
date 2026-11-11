# Neural Network Surrogate for an Adiabatic Plug Flow Reactor (PFR)

Python Programming project — SRM Institute of Science and Technology
**Team members:** ANISH RAVIKUMAR (RA2411007010026), VAITHEESHVARAN S (RA2411007010020), SHREE KARTHIKEYAN K M (RA2411007010038)

## Overview
An adiabatic PFR with an exothermic, second-order reaction (A → products) is described by two coupled ODEs
for concentration `C_A` and temperature `T`. This project:

1. solves the ODEs numerically with `scipy.integrate.solve_ivp` for 9000 operating conditions
   (inlet concentration `C_A0`, inlet temperature `T0`, residence time `tau`) chosen by Latin Hypercube Sampling;
2. trains a fully connected neural network (PyTorch) to predict the exit values `C_A_exit` and `T_exit`
   directly from `(C_A0, T0, tau)`;
3. evaluates the surrogate on unseen test data (R², RMSE), benchmarks its speed against `solve_ivp`,
   and saves parity plots.

## Model equations
Rate: `r = K0 * exp(-Ea / (R*T)) * C_A^2`
`dC_A/dt = -r`  `dT/dt = (-ΔH) * r / (rho*cp)`
Constants: `K0 = 5e6`, `Ea = 65000 J/mol`, `ΔH = -75000 J/mol`, `rho*cp = 1000`, `R = 8.314`.

## Input ranges
| Variable | Range |
|---|---|
| C_A0 | 0.5 – 5.0 |
| T0 (K) | 300 – 400 |
| tau (s) | 1 – 60 |

## Requirements
Python 3.9+ and the packages in `requirements.txt`.

```
pip install -r requirements.txt
```

## How to run
```
python pfr.py
```
The script prints the validation loss every 25 epochs, the test R² and RMSE for both outputs, and the
speed comparison, then writes `parity.png`. Data generation (9000 ODE solves) takes a few minutes on a CPU.
Random seeds are fixed (42), but results can still differ slightly between machines.

## Files
| File | Description |
|---|---|
| `pfr.py` | Full pipeline: data generation, training, evaluation, benchmark, plots |
| `parity.png` | Parity plots (true vs predicted) for the test set — created when the script runs |
| `output_log.txt` | Terminal output of one run (validation loss, test R² and RMSE, speed comparison) |
| `requirements.txt` | Python dependencies |

## Method summary
- Data split 70 / 15 / 15 (train / validation / test); inputs and outputs standardised using training-set statistics only.
- Network: 3 → 4 hidden layers of 64 units (GELU) → 2 outputs; Adam, lr 1e-3, batch size 512, up to 350 epochs.
- Early stopping on validation MSE with patience 50; the best-validation weights are restored.

## Results (one run)
| Output | R² | RMSE |
|---|---|---|
| C_A_exit | 0.998961 | 0.04811 |
| T_exit | 0.999406 | 3.589 K |

Speed for 1000 cases: neural network 0.0030 s, `solve_ivp` 5.16 s (about 1700×).
The network is evaluated in one batched call, while the ODE solver runs the cases one at a time, so the speed-up
partly reflects batching. Results can differ slightly between machines.

## Limitations
- Valid only inside the sampled input ranges.
- A few outlying predictions; some predicted `C_A_exit` values slightly below zero.
- Single run and single random seed; no baseline model compared.
