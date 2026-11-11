import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from scipy.integrate import solve_ivp
from scipy.stats import qmc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

np.random.seed(42); torch.manual_seed(42)

R, K0, EA, DH, RHOCP = 8.314, 5e6, 65000.0, -75000.0, 1000.0

def odes(t, y):
    ca, T = y
    r = K0 * np.exp(-EA / (R * T)) * ca**2
    return [-r, -DH * r / RHOCP]

def solve(ca0, T0, tau):
    s = solve_ivp(odes, (0, tau), [ca0, T0], rtol=1e-8, atol=1e-10)
    return s.y[0, -1], s.y[1, -1]

# 1. Generate data (Latin hypercube over C_A0, T0, tau)
X = qmc.scale(qmc.LatinHypercube(d=3, seed=42).random(9000),
              [0.5, 300, 1], [5.0, 400, 60])
Y = np.array([solve(*x) for x in X])

# 2. Split (70/15/15) and scale
Xtr, Xt, Ytr, Yt = train_test_split(X, Y, test_size=0.3, random_state=42)
Xv, Xte, Yv, Yte = train_test_split(Xt, Yt, test_size=0.5, random_state=42)
xs, ys = StandardScaler().fit(Xtr), StandardScaler().fit(Ytr)
T = lambda a, s: torch.tensor(s.transform(a), dtype=torch.float32)
Xtr_t, Ytr_t, Xv_t, Yv_t, Xte_t = T(Xtr, xs), T(Ytr, ys), T(Xv, xs), T(Yv, ys), T(Xte, xs)

# 3. Model
model = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(),
                      nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(),
                      nn.Linear(64, 2))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
best, best_state, wait = 1e9, None, 0

# 4. Train with early stopping
for ep in range(350):
    perm = torch.randperm(len(Xtr_t))
    for i in range(0, len(perm), 512):
        b = perm[i:i + 512]
        opt.zero_grad()
        nn.functional.mse_loss(model(Xtr_t[b]), Ytr_t[b]).backward()
        opt.step()
    with torch.no_grad():
        v = nn.functional.mse_loss(model(Xv_t), Yv_t).item()
    if v < best - 1e-7:
        best, wait = v, 0
        best_state = {k: p.clone() for k, p in model.state_dict().items()}
    else:
        wait += 1
        if wait >= 50: break
    if ep % 25 == 0: print(f"epoch {ep} val MSE {v:.3e}")
model.load_state_dict(best_state)

# 5. Test metrics
with torch.no_grad():
    pred = ys.inverse_transform(model(Xte_t).numpy())
for j, n in enumerate(["C_A_exit", "T_exit"]):
    print(f"{n}: R2={r2_score(Yte[:, j], pred[:, j]):.6f}  "
          f"RMSE={mean_squared_error(Yte[:, j], pred[:, j])**0.5:.4g}")

# 6. Speed benchmark (1000 cases)
Xb = Xte[:1000]
t0 = time.perf_counter()
with torch.no_grad(): ys.inverse_transform(model(T(Xb, xs)).numpy())
t_nn = time.perf_counter() - t0
t0 = time.perf_counter()
for x in Xb: solve(*x)
t_ode = time.perf_counter() - t0
print(f"NN {t_nn:.4f}s | solve_ivp {t_ode:.2f}s | speedup {t_ode/t_nn:.0f}x")

# 7. Parity plots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for j, n in enumerate(["C_A_exit", "T_exit"]):
    ax[j].scatter(Yte[:, j], pred[:, j], s=5, alpha=0.4)
    ax[j].plot(Yte[:, j], Yte[:, j], "k--")
    ax[j].set(xlabel=f"True {n}", ylabel=f"Predicted {n}")
plt.tight_layout(); plt.savefig("parity.png", dpi=150)