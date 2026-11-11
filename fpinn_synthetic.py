"""
fpinn_synthetic.py  -  Section 6.1 (Synthetic data) for the paper
"Fractional Physics-Informed Neural Networks with Time-Varying Transmission
 for Modelling and Forecasting COVID-19 Dynamics"  (Waseem Shokat)

What it does
------------
1. Simulates the corrected fractional model (3.1) with the L1 scheme for known
   memory orders z* and a known transmission rate alpha*(tau) that changes with
   a synthetic lockdown/reopening (stringency + mobility covariates).
2. Adds negative-binomial noise to daily cases and deaths.
3. Trains the fPINN of Section 4 (state network, transmission network,
   trainable z and reporting fraction varpi; L1 physics residuals; cumulative
   cases/deaths from the Riemann-Liouville integral; staged training).
4. Runs the four experiments of Section 6.1:
      E1 recovery of z and alpha  (z* = 0.8, 0.9, 1.0; low / high noise)
      E2 figures (learned vs true alpha, fitted cases)
      E3 identifiability (z fixed at 1 vs z learned)
      E4 ablation (transmission network with vs without covariates)
5. Writes LaTeX tables, PDF figures, a CSV and a text summary to ./results/

Usage
-----
    pip install torch numpy scipy matplotlib
    python fpinn_synthetic.py --quick      # fast test run (few minutes)
    python fpinn_synthetic.py              # full run for the paper

Parameters: Arshad et al. (2026), Table 1, except kappa, gamma, rho_I and rho_D,
which are replaced by realistic values (case fatality about 1-2 %).
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

# ----------------------------------------------------------------------------
# 1. Model parameters (per day).  >>> Replace with your paper's values <<<
# ----------------------------------------------------------------------------
PARAMS = dict(
    # Arshad et al. (2026), Table 1:
    delta=0.09564,  # relative infectiousness of asymptomatic
    psi=0.13100,    # rate of leaving the exposed class
    eta=0.76922,    # fraction symptomatic
    mu_A=0.2,       # diagnosis rate, asymptomatic
    mu_I=0.25,      # diagnosis rate, symptomatic
    phi_A=0.14960,  # recovery, asymptomatic
    phi_I=0.0998,   # recovery, symptomatic
    phi_D=0.14960,  # recovery, diagnosed
    sigma=0.005,    # vaccine inefficacy
    # replaced by realistic values (case fatality ~1-2 %, short quarantine):
    kappa=0.005,    # quarantine rate (Arshad: 0.2914)
    gamma=0.05,     # release from quarantine, ~20 days (Arshad: 0.01796)
    rho_I=0.002,    # disease death, symptomatic (Arshad: 0.20806)
    rho_D=0.005,    # disease death, diagnosed (Arshad: 0.29685)
    # natural death rate (life expectancy ~70 years)
    d=1 / (70 * 365),
)
NAMES = ["S", "Q", "E", "A", "I", "D", "R", "V"]

T_TOTAL = 200          # days simulated
T_TRAIN = 160          # days used for training (rest = test period)
N_POP = 1.0e7          # population size for counts
VARPI_TRUE = 0.4       # true reporting fraction
# baseline transmission rate for each true z (chosen so that the epidemic curves
# have a similar shape: one wave damped by the lockdown, a second wave after reopening)
ALPHA0_BY_Z = {0.8: 1.9, 0.9: 1.5, 1.0: 1.25}
H_SIM = 0.1            # step of the data-generating simulation (days)
H_PINN = 1.0           # step of the fPINN collocation grid (days)


# ----------------------------------------------------------------------------
# 2. Synthetic covariates, alpha*(tau) and beta(tau)
# ----------------------------------------------------------------------------
def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def stringency(t):
    """Lockdown: 0 -> 0.8 (days 30-40), hold, partial reopening to 0.3 (days 90-120)."""
    t = np.asarray(t, dtype=float)
    up = 0.8 * smoothstep((t - 30) / 10)
    down = 0.5 * smoothstep((t - 90) / 30)
    return up - down


def mobility(t):
    """Mobility falls when stringency rises (rescaled to [0, 1])."""
    return 1.0 - 0.7 * stringency(t)


def alpha_true(t, a0):
    return a0 * (1.0 - 0.6 * stringency(t))


def beta_fun(t):
    """Vaccination starts on day 100 at 0.2 % of susceptibles per day."""
    t = np.asarray(t, dtype=float)
    return 0.002 * smoothstep((t - 100) / 5)


# ----------------------------------------------------------------------------
# 3. Right-hand side of model (3.1), numpy and torch versions
#    (rates are raised to the power z, Lambda^z = d^z * N0 with N0 = 1)
# ----------------------------------------------------------------------------
def rhs_np(X, a, b, z, p):
    S, Q, E, A, I, D, R, V = X[:8]
    P = {k: v ** z for k, v in p.items()}
    az, bz = a ** z, b ** z
    Lam = P["d"]
    force = I + P["delta"] * A
    k1 = P["mu_A"] + P["phi_A"] + P["d"]
    k2 = P["mu_I"] + P["phi_I"] + P["rho_I"] + P["d"]
    dS = Lam - az * S * force - (P["kappa"] + bz + P["d"]) * S + P["gamma"] * Q
    dQ = P["kappa"] * S - (P["gamma"] + P["d"]) * Q
    dE = az * (S + P["sigma"] * V) * force - (P["psi"] + P["d"]) * E
    dA = P["psi"] * (1 - p["eta"] ** z) * E - k1 * A
    dI = P["psi"] * P["eta"] * E - k2 * I
    dD = P["mu_A"] * A + P["mu_I"] * I - (P["phi_D"] + P["rho_D"] + P["d"]) * D
    dR = P["phi_A"] * A + P["phi_I"] * I + P["phi_D"] * D - P["d"] * R
    dV = bz * S - P["sigma"] * az * V * force - P["d"] * V
    dC = P["mu_A"] * A + P["mu_I"] * I                 # cumulative diagnoses
    dM = P["rho_I"] * I + P["rho_D"] * D               # cumulative deaths
    return np.array([dS, dQ, dE, dA, dI, dD, dR, dV, dC, dM])


def rhs_torch(X, a, b, z, p):
    """X: (n, 8) tensor, a, b: (n,) tensors, z: scalar tensor."""
    S, Q, E, A, I, D, R, V = [X[:, i] for i in range(8)]
    P = {k: torch.as_tensor(v) ** z for k, v in p.items()}
    az, bz = a ** z, b ** z
    Lam = P["d"]
    force = I + P["delta"] * A
    k1 = P["mu_A"] + P["phi_A"] + P["d"]
    k2 = P["mu_I"] + P["phi_I"] + P["rho_I"] + P["d"]
    dS = Lam - az * S * force - (P["kappa"] + bz + P["d"]) * S + P["gamma"] * Q
    dQ = P["kappa"] * S - (P["gamma"] + P["d"]) * Q
    dE = az * (S + P["sigma"] * V) * force - (P["psi"] + P["d"]) * E
    dA = P["psi"] * (1 - P["eta"]) * E - k1 * A
    dI = P["psi"] * P["eta"] * E - k2 * I
    dD = P["mu_A"] * A + P["mu_I"] * I - (P["phi_D"] + P["rho_D"] + P["d"]) * D
    dR = P["phi_A"] * A + P["phi_I"] * I + P["phi_D"] * D - P["d"] * R
    dV = bz * S - P["sigma"] * az * V * force - P["d"] * V
    return torch.stack([dS, dQ, dE, dA, dI, dD, dR, dV], dim=1)


# ----------------------------------------------------------------------------
# 4. Explicit L1 solver (used for data generation and for forecasting)
# ----------------------------------------------------------------------------
def l1_weights(n, z):
    k = np.arange(n)
    return (k + 1) ** (1 - z) - k ** (1 - z)


def solve_l1(z, h, t_end, alpha_f, X0=None, history=None, t0=0.0):
    """Explicit L1 scheme for the augmented system (8 classes + C + M).
    history: optional array (m+1, 10) of the solution on the grid t0-m*h..t0
             (used to continue a trajectory, i.e. for forecasting with memory)."""
    c = h ** (-z) / math.gamma(2 - z)
    if history is None:
        hist = [np.asarray(X0, dtype=float)]
        t_start = 0.0
    else:
        hist = [row for row in np.asarray(history, dtype=float)]
        t_start = t0 - (len(hist) - 1) * h
    n_total = int(round((t_end - t_start) / h))
    b = l1_weights(n_total + 1, z)
    X = np.array(hist)
    dX = np.diff(X, axis=0) if len(X) > 1 else np.zeros((0, 10))
    out = list(X)
    incs = list(dX)
    for n in range(len(out), n_total + 1):
        t_prev = t_start + (n - 1) * h
        F = rhs_np(out[-1], alpha_f(t_prev), beta_fun(t_prev), z, PARAMS)
        mem = 0.0
        if n > 1:
            past = np.array(incs[::-1])              # dX_{n-1}, dX_{n-2}, ...
            mem = (b[1:n, None] * past[: n - 1]).sum(axis=0)
        inc = F / c - mem
        new = np.maximum(out[-1] + inc, 0.0)
        incs.append(new - out[-1])
        out.append(new)
    t = t_start + h * np.arange(len(out))
    return t, np.array(out)


def initial_state():
    E0, A0, I0 = 2e-5, 1e-5, 5e-6
    X0 = np.zeros(10)
    X0[0] = 1.0 - E0 - A0 - I0
    X0[2], X0[3], X0[4] = E0, A0, I0
    return X0


def generate_data(z_true, dispersion, seed):
    """Returns dict with daily true/noisy cases and deaths and the daily states."""
    rng = np.random.default_rng(seed)
    a0 = ALPHA0_BY_Z[z_true]
    t, X = solve_l1(z_true, H_SIM, T_TOTAL, lambda s: alpha_true(s, a0), X0=initial_state())
    step = int(round(1.0 / H_SIM))
    Xd = X[::step]                                   # daily values, days 0..T
    C, M = Xd[:, 8], Xd[:, 9]
    mean_y = VARPI_TRUE * N_POP * np.maximum(np.diff(C), 0)
    mean_d = N_POP * np.maximum(np.diff(M), 0)

    def nb(mean):
        mean = np.maximum(mean, 1e-8)
        pr = dispersion / (dispersion + mean)
        return rng.negative_binomial(dispersion, pr).astype(float)

    return dict(days=np.arange(0, T_TOTAL + 1), X=Xd, y=nb(mean_y), d=nb(mean_d),
                y_true=mean_y, d_true=mean_d, a0=a0, z_true=z_true)


# ----------------------------------------------------------------------------
# 5. fPINN (Section 4)
# ----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, n_in, n_out, width, depth):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FPINN(nn.Module):
    def __init__(self, use_covariates=True, learn_z=True, z_fixed=None, z_init=0.9,
                 width=64, depth=4, a_width=32, a_depth=2):
        super().__init__()
        self.state = MLP(1, 8, width, depth)
        # offset so that the initial outputs are of a sensible size (log scale)
        self.register_buffer("log_off", torch.log(torch.tensor(
            [0.99, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4, 1e-3, 1e-4])))
        self.use_cov = use_covariates
        self.alpha_net = MLP(3 if use_covariates else 1, 1, a_width, a_depth)
        with torch.no_grad():                      # softplus(out) ~ 0.5 at start
            self.alpha_net.net[-1].bias.fill_(math.log(math.exp(0.5) - 1))
        self.learn_z = learn_z
        if learn_z:
            u = (z_init - 0.5) / 0.5
            self.zeta = nn.Parameter(torch.tensor(math.log(u / (1 - u))))
        else:
            self.register_buffer("z_fix", torch.tensor(float(z_fixed)))
        self.theta = nn.Parameter(torch.tensor(0.0))   # varpi = sigmoid(theta)

    def z(self):
        return 0.5 + 0.5 * torch.sigmoid(self.zeta) if self.learn_z else self.z_fix

    def varpi(self):
        return torch.sigmoid(self.theta)

    def states(self, tau_hat):
        return torch.exp(self.state(tau_hat[:, None]) + self.log_off)

    def alpha(self, tau_hat, m, s):
        inp = torch.stack([tau_hat, m, s], 1) if self.use_cov else tau_hat[:, None]
        return nn.functional.softplus(self.alpha_net(inp))[:, 0]


def l1_matrix(N, z):
    """Lower-triangular Toeplitz matrix B(z) with B[n-1, m-1] = b_{n-m}, m <= n."""
    n = torch.arange(1, N + 1, dtype=torch.float64)
    K = n[:, None] - n[None, :]
    mask = K >= 0
    Kc = torch.clamp(K, min=0)
    B = (Kc + 1) ** (1 - z) - Kc ** (1 - z)
    return B * mask


def rl_matrix(N, z):
    """W[n-1, k] = (n-k)^z - (n-k-1)^z for k = 0..n-1 (product rectangle rule)."""
    n = torch.arange(1, N + 1, dtype=torch.float64)[:, None]
    k = torch.arange(0, N, dtype=torch.float64)[None, :]
    J = n - 1 - k
    mask = J >= 0
    Jc = torch.clamp(J, min=0)
    W = (Jc + 1) ** z - Jc ** z
    return W * mask


def forward_quantities(model, days, cov, h=H_PINN):
    """Compute states, alpha, residuals, cumulative C and M on the training grid."""
    T = days[-1]
    tau = torch.as_tensor(days, dtype=torch.float64)
    tau_hat = tau / T
    m = torch.as_tensor(cov["m"]); s = torch.as_tensor(cov["s"]); b = torch.as_tensor(cov["b"])
    X = model.states(tau_hat)                       # (N+1, 8)
    a = model.alpha(tau_hat, m, s)                  # (N+1,)
    z = model.z()
    N = len(days) - 1
    c = h ** (-z) / torch.exp(torch.lgamma(2 - z))
    dX = X[1:] - X[:-1]
    DzX = c * (l1_matrix(N, z) @ dX)               # L1 Caputo derivative at tau_1..tau_N
    F = rhs_torch(X[1:], a[1:], b[1:], z, PARAMS)
    scale = X.detach().mean(0) + 1e-12
    resid = (DzX - F) / scale
    P = {k: torch.as_tensor(v) ** z for k, v in PARAMS.items()}
    inflow_C = P["mu_A"] * X[:, 3] + P["mu_I"] * X[:, 4]
    inflow_M = P["rho_I"] * X[:, 4] + P["rho_D"] * X[:, 5]
    W = rl_matrix(N, z) * (h ** z / torch.exp(torch.lgamma(z + 1)))
    C = torch.cat([torch.zeros(1), W @ inflow_C[:-1]])
    M = torch.cat([torch.zeros(1), W @ inflow_M[:-1]])
    return dict(X=X, a=a, z=z, resid=resid, C=C, M=M)


def train_fpinn(data, cov, seed, use_cov=True, learn_z=True, z_fixed=None,
                iters=(1500, 3000, 3000), lam_phys=100.0, lam_pop=1.0, lam_s=0.01,
                w_d=1.0, lr=2e-3, lr_scalar=2e-2, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    z_init = float(np.random.uniform(0.75, 0.97))  # random start, tests sensitivity
    model = FPINN(use_covariates=use_cov, learn_z=learn_z, z_fixed=z_fixed, z_init=z_init)
    # the scalar parameters (z, varpi) get a larger learning rate than the networks
    scal = [p for n, p in model.named_parameters() if n in ("zeta", "theta")]
    nets = [p for n, p in model.named_parameters() if n not in ("zeta", "theta")]
    opt = torch.optim.Adam([{"params": nets, "lr": lr},
                            {"params": scal, "lr": lr_scalar}])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=sum(iters) // 3, gamma=0.5)
    days = data["days_train"]
    y = torch.as_tensor(data["y_train"]); d = torch.as_tensor(data["d_train"])
    n1, n2, n3 = iters
    for it in range(n1 + n2 + n3):
        lp = 0.0 if it < n1 else (lam_phys * min(1.0, (it - n1) / max(n2, 1)))
        q = forward_quantities(model, days, cov)
        yhat = model.varpi() * N_POP * (q["C"][1:] - q["C"][:-1])
        dhat = N_POP * (q["M"][1:] - q["M"][:-1])
        L_data = ((torch.log1p(torch.clamp(yhat, min=0)) - torch.log1p(y)) ** 2
                  + w_d * (torch.log1p(torch.clamp(dhat, min=0)) - torch.log1p(d)) ** 2).mean()
        L_phys = (q["resid"] ** 2).mean()
        L_pop = (q["X"][0].sum() - 1.0) ** 2
        L_s = ((q["a"][1:] - q["a"][:-1]) ** 2).mean()
        loss = L_data + lp * L_phys + lam_pop * L_pop + lam_s * L_s
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and it % 500 == 0:
            print(f"  it {it:5d}  loss {loss.item():.3e}  data {L_data.item():.3e} "
                  f"phys {L_phys.item():.3e}  z {q['z'].item():.3f}")
    with torch.no_grad():
        q = forward_quantities(model, days, cov)
        yhat = model.varpi() * N_POP * (q["C"][1:] - q["C"][:-1])
    return model, dict(z=float(q["z"].detach()), varpi=float(model.varpi().detach()),
                       alpha=q["a"].numpy(), X=q["X"].numpy(), C=q["C"].numpy(),
                       M=q["M"].numpy(), yhat=yhat.numpy(), L_data=float(L_data))


def forecast(model, res, data, cov_full):
    """Continue the learned trajectory with the explicit L1 solver (memory kept)."""
    z = res["z"]
    hist = np.concatenate([res["X"], res["C"][:, None], res["M"][:, None]], axis=1)
    tau_all = torch.as_tensor(data["days"], dtype=torch.float64)

    def a_learned(t):
        with torch.no_grad():
            th = torch.tensor([t / data["days_train"][-1]])
            m = torch.tensor([float(mobility(t))]); s = torch.tensor([float(stringency(t))])
            return float(model.alpha(th, m, s)[0].detach())

    t, Xf = solve_l1(z, H_PINN, T_TOTAL, a_learned, history=hist,
                     t0=float(data["days_train"][-1]))
    C = Xf[:, 8]
    yf = res["varpi"] * N_POP * np.diff(C)
    return yf[T_TRAIN:]                              # forecasts for days T_TRAIN+1..T_TOTAL


# ----------------------------------------------------------------------------
# 6. Metrics
# ----------------------------------------------------------------------------
def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def wis(y, samples, alphas=(0.02, 0.05, 0.1, 0.2, 0.5)):
    """Weighted interval score (Bracher et al. 2021) from ensemble samples."""
    med = np.median(samples, axis=0)
    score = 0.5 * np.abs(y - med)
    for a in alphas:
        lo = np.quantile(samples, a / 2, axis=0); hi = np.quantile(samples, 1 - a / 2, axis=0)
        IS = (hi - lo) + (2 / a) * (lo - y) * (y < lo) + (2 / a) * (y - hi) * (y > hi)
        score += (a / 2) * IS
    return float(np.mean(score / (len(alphas) + 0.5)))


# ----------------------------------------------------------------------------
# 7. Experiments
# ----------------------------------------------------------------------------
def prepare(data):
    data["days_train"] = data["days"][: T_TRAIN + 1]
    data["y_train"] = data["y"][:T_TRAIN]
    data["d_train"] = data["d"][:T_TRAIN]
    return data


def cov_on(days):
    return dict(m=mobility(days), s=stringency(days), b=beta_fun(days))


def run_ensemble(data, K, iters, **kw):
    cov = cov_on(data["days_train"])
    runs = []
    for k in range(K):
        t0 = time.time()
        model, res = train_fpinn(data, cov, seed=1000 + k, iters=iters, **kw)
        res["forecast"] = forecast(model, res, data, None)
        res["time"] = time.time() - t0
        runs.append(res)
        print(f"    seed {k}: z = {res['z']:.3f}, varpi = {res['varpi']:.3f}, "
              f"{res['time']:.0f} s")
    return runs


def summarise(runs, data):
    zs = np.array([r["z"] for r in runs])
    a_true = alpha_true(data["days_train"].astype(float), data["a0"])
    a_err = np.array([rel_l2(r["alpha"], a_true) for r in runs])
    y_test = data["y"][T_TRAIN:]
    F = np.array([r["forecast"] for r in runs])
    med = np.median(F, axis=0)
    lo, hi = np.quantile(F, 0.025, axis=0), np.quantile(F, 0.975, axis=0)
    return dict(z_med=float(np.median(zs)), z_lo=float(np.quantile(zs, 0.025)),
                z_hi=float(np.quantile(zs, 0.975)), a_err=float(np.median(a_err)),
                mae=float(np.mean(np.abs(med - y_test))),
                rmse=float(np.sqrt(np.mean((med - y_test) ** 2))),
                coverage=float(np.mean((y_test >= lo) & (y_test <= hi))),
                wis=wis(y_test, F), L_data=float(np.median([r["L_data"] for r in runs])))


def fmt(x, nd=3):
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast test run")
    ap.add_argument("--K", type=int, default=None, help="ensemble size")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    K = args.K or (2 if args.quick else 10)
    iters = (150, 300, 300) if args.quick else (1500, 3000, 3000)
    z_list = [0.8, 0.9, 1.0]
    noise = {"low": 100.0, "high": 10.0}            # NB dispersion (higher = less noise)
    rows, store = [], {}

    # E1 + E2: recovery
    for zt in z_list:
        for nl, disp in noise.items():
            print(f"E1: z* = {zt}, noise = {nl}")
            data = prepare(generate_data(zt, disp, seed=int(zt * 100) + int(disp)))
            runs = run_ensemble(data, K, iters)
            s = summarise(runs, data)
            rows.append(dict(exp="E1", z_true=zt, noise=nl, **s))
            store[(zt, nl)] = (data, runs)

    # E3: identifiability (z fixed at 1 vs learned), z* = 0.8, low noise
    data, runs_learned = store[(0.8, "low")]
    print("E3: z fixed at 1")
    runs_fixed = run_ensemble(data, K, iters, learn_z=False, z_fixed=0.9999)
    s_fix = summarise(runs_fixed, data); s_learn = summarise(runs_learned, data)
    rows.append(dict(exp="E3_fixed_z1", z_true=0.8, noise="low", **s_fix))
    rows.append(dict(exp="E3_learned", z_true=0.8, noise="low", **s_learn))

    # E4: ablation without covariates, z* = 0.9, low noise
    data9, runs_cov = store[(0.9, "low")]
    print("E4: no covariates")
    runs_nocov = run_ensemble(data9, K, iters, use_cov=False)
    s_nocov = summarise(runs_nocov, data9); s_cov = summarise(runs_cov, data9)
    rows.append(dict(exp="E4_no_cov", z_true=0.9, noise="low", **s_nocov))
    rows.append(dict(exp="E4_cov", z_true=0.9, noise="low", **s_cov))

    # ---------------- outputs ----------------
    import csv
    with open(os.path.join(args.out, "results_synthetic.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Table 6.1 (recovery)
    lines = []
    for r in [r for r in rows if r["exp"] == "E1"]:
        lines.append(f"{r['z_true']:.1f} & {r['noise']} & {fmt(r['z_med'])} "
                     f"({fmt(r['z_lo'])}, {fmt(r['z_hi'])}) & {fmt(r['a_err'])} \\\\")
    with open(os.path.join(args.out, "table_synthetic_rows.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # Identifiability + ablation table
    def row(label, s):
        return (f"{label} & {fmt(s['z_med'])} & {fmt(s['L_data'], 4)} & "
                f"{fmt(s['a_err'])} & {s['mae']:.1f} & {s['coverage']*100:.0f}\\% \\\\")
    with open(os.path.join(args.out, "table_ident_ablation.tex"), "w") as f:
        f.write("\\begin{tabular}{lccccc}\n\\toprule\n"
                "Setting & $\\hat z$ & Data loss & $\\alpha$ rel.\\ error & Test MAE & 95\\% coverage \\\\\n"
                "\\midrule\n")
        f.write(row("$z$ fixed at 1 ($z^\\ast=0.8$)", s_fix) + "\n")
        f.write(row("$z$ learned ($z^\\ast=0.8$)", s_learn) + "\n")
        f.write("\\midrule\n")
        f.write(row("No covariates ($z^\\ast=0.9$)", s_nocov) + "\n")
        f.write(row("With covariates ($z^\\ast=0.9$)", s_cov) + "\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # Figure: learned vs true alpha (low noise), one panel per z*
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, zt in zip(axes, z_list):
        data, runs = store[(zt, "low")]
        t = data["days_train"]
        A = np.array([r["alpha"] for r in runs])
        ax.fill_between(t, np.quantile(A, 0.025, 0), np.quantile(A, 0.975, 0),
                        color="tab:blue", alpha=0.25, label="95% band")
        ax.plot(t, np.median(A, 0), color="tab:blue", lw=1.8, label="learned (median)")
        ax.plot(t, alpha_true(t.astype(float), data["a0"]), "k--", lw=1.4, label="true")
        ax.set_title(f"$z^*={zt}$"); ax.set_xlabel("day")
    axes[0].set_ylabel(r"$\alpha(\tau)$"); axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "fig_synthetic_alpha.pdf"))

    # Figure: fitted and forecast cases (low noise)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=False)
    for ax, zt in zip(axes, z_list):
        data, runs = store[(zt, "low")]
        dd = data["days"][1:]
        ax.plot(dd, data["y"], ".", ms=3, color="0.5", label="synthetic data")
        Yfit = np.array([r["yhat"] for r in runs])
        ax.plot(dd[:T_TRAIN], np.median(Yfit, 0), color="tab:blue", lw=1.6, label="fit")
        Ff = np.array([r["forecast"] for r in runs])
        ax.plot(dd[T_TRAIN:], np.median(Ff, 0), color="tab:red", lw=1.6, label="forecast")
        ax.fill_between(dd[T_TRAIN:], np.quantile(Ff, 0.025, 0), np.quantile(Ff, 0.975, 0),
                        color="tab:red", alpha=0.25)
        ax.axvline(T_TRAIN, color="k", lw=0.8, ls=":")
        ax.set_title(f"$z^*={zt}$"); ax.set_xlabel("day")
    axes[0].set_ylabel("daily reported cases"); axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "fig_synthetic_cases.pdf"))

    # Text summary for writing Section 6.1
    e1 = [r for r in rows if r["exp"] == "E1"]
    inside = sum(1 for r in e1 if r["z_lo"] <= r["z_true"] <= r["z_hi"])
    max_err = max(abs(r["z_med"] - r["z_true"]) for r in e1)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(f"Ensemble size K = {K}, iterations = {iters}\n")
        f.write(f"True z inside 95% interval: {inside} of {len(e1)} cases\n")
        f.write(f"Largest |median z - true z|: {max_err:.3f}\n")
        for r in e1:
            f.write(f"z*={r['z_true']}, noise={r['noise']}: z_hat={r['z_med']:.3f} "
                    f"[{r['z_lo']:.3f}, {r['z_hi']:.3f}], alpha rel err={r['a_err']:.3f}, "
                    f"test MAE={r['mae']:.1f}, coverage={r['coverage']*100:.0f}%, WIS={r['wis']:.1f}\n")
        f.write(f"E3 fixed z=1: data loss {s_fix['L_data']:.4f}, test MAE {s_fix['mae']:.1f}, "
                f"alpha err {s_fix['a_err']:.3f}\n")
        f.write(f"E3 learned z: data loss {s_learn['L_data']:.4f}, test MAE {s_learn['mae']:.1f}, "
                f"alpha err {s_learn['a_err']:.3f}\n")
        f.write(f"E4 no covariates: alpha err {s_nocov['a_err']:.3f}, z [{s_nocov['z_lo']:.3f}, "
                f"{s_nocov['z_hi']:.3f}], MAE {s_nocov['mae']:.1f}\n")
        f.write(f"E4 covariates:    alpha err {s_cov['a_err']:.3f}, z [{s_cov['z_lo']:.3f}, "
                f"{s_cov['z_hi']:.3f}], MAE {s_cov['mae']:.1f}\n")
    with open(os.path.join(args.out, "settings.json"), "w") as f:
        json.dump(dict(PARAMS=PARAMS, T_TOTAL=T_TOTAL, T_TRAIN=T_TRAIN, N_POP=N_POP,
                       VARPI_TRUE=VARPI_TRUE, ALPHA0_BY_Z={str(k): v for k, v in ALPHA0_BY_Z.items()},
                       H_SIM=H_SIM, H_PINN=H_PINN,
                       K=K, iters=iters, noise=noise, lam_phys=100.0, lam_s=0.01), f, indent=2)
    print("\nDone. Files written to", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
