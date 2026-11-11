"""
fpinn_turkey.py  -  Section 6.2 (Turkiye, 1 Dec 2020 - 30 Jun 2021)

Uses the model, fPINN and L1 solver of fpinn_synthetic.py, applied to real data:
  * daily cases and deaths: JHU CSSE time series (Turkey)
  * first vaccine doses: Our World in Data (vaccinations.csv)
  * stringency: OxCGRT StringencyIndex_Average (national total)
  * mobility: Google Community Mobility Reports (mean of retail & recreation,
    transit stations and workplaces)

Models compared (each an ensemble of K networks):
  M1  fPINN: learned z, alpha(tau) from time + mobility + stringency
  M2  same, z fixed at 1 (integer-order PINN)
  M3  learned z, constant alpha (constant-parameter fractional model)

Usage:  python fpinn_turkey.py --data ../data_tr --out results_tr [--quick]
"""
import argparse, json, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fpinn_synthetic as F

START, END = "2020-12-01", "2021-06-30"
H_TEST = 40
# Death rates: the synthetic values (rho_I=0.002, rho_D=0.005) imply that about
# 4 % of diagnosed people die, which would need a reporting fraction above 1 for
# Turkiye.  We use smaller rates (about 0.4 % of diagnosed people die).
F.PARAMS["rho_I"] = 0.0001
F.PARAMS["rho_D"] = 0.0006


def load(data_dir):
    rng = pd.date_range(START, END)
    ext = pd.date_range(pd.Timestamp(START) - pd.Timedelta(days=10), END)

    def jhu(fn):
        j = pd.read_csv(os.path.join(data_dir, fn))
        r = j[(j["Country/Region"] == "Turkey") & j["Province/State"].isna()].iloc[0, 4:]
        r.index = pd.to_datetime(r.index, format="%m/%d/%y")
        return r.astype(float).diff()

    cases, deaths = jhu("jhu_cases.csv"), jhu("jhu_deaths.csv")
    # 10 Dec 2020: one-off report of 823,225 earlier asymptomatic cases (Jul-Nov
    # 2020); replace by the mean of the neighbouring days
    day = pd.Timestamp("2020-12-10")
    raw_dump = cases[day]
    cases[day] = 0.5 * (cases[day - pd.Timedelta(days=1)] + cases[day + pd.Timedelta(days=1)])
    cases, deaths = cases.clip(lower=0), deaths.clip(lower=0)
    cases = cases.rolling(7).mean().reindex(ext)
    deaths = deaths.rolling(7).mean().reindex(ext)

    v = pd.read_csv(os.path.join(data_dir, "vacc.csv"))
    v = v[v.iso_code == "TUR"].set_index("date"); v.index = pd.to_datetime(v.index)
    first = v["people_vaccinated"].copy()
    early = v.index < pd.Timestamp("2021-02-11")          # second doses start ~11 Feb
    first[early & first.isna()] = v.loc[early, "total_vaccinations"]
    first = first.reindex(pd.date_range("2020-12-01", "2021-07-10")).interpolate().fillna(0)
    doses = first.diff().clip(lower=0).rolling(7).mean().reindex(rng).fillna(0)

    o = pd.read_csv(os.path.join(data_dir, "oxcgrt.csv"),
                    usecols=["CountryCode", "Jurisdiction", "Date", "StringencyIndex_Average"])
    o = o[(o.CountryCode == "TUR") & (o.Jurisdiction == "NAT_TOTAL")]
    o.index = pd.to_datetime(o.Date.astype(str), format="%Y%m%d")
    s = (o.StringencyIndex_Average.reindex(rng).interpolate() / 100.0)

    m = pd.concat([pd.read_csv(os.path.join(data_dir, f"{y}_TR_Region_Mobility_Report.csv"))
                   for y in (2020, 2021)])
    m = m[m.sub_region_1.isna()].copy(); m.index = pd.to_datetime(m.date)
    cols = ["retail_and_recreation_percent_change_from_baseline",
            "transit_stations_percent_change_from_baseline",
            "workplaces_percent_change_from_baseline"]
    mob = m[cols].mean(axis=1).rolling(7).mean().reindex(rng).interpolate()
    mob = (mob - mob.min()) / (mob.max() - mob.min())

    pop = float(v.shape[0] and 85341248)                  # OWID population for Turkiye
    return dict(dates=rng, cases=cases.reindex(rng).values, deaths=deaths.reindex(rng).values,
                doses=doses.values, s=s.values, m=mob.values, pop=pop, raw_dump=float(raw_dump))


class ConstAlphaFPINN(F.FPINN):
    """Constant transmission rate (one trainable scalar)."""
    def __init__(self, **kw):
        kw["use_covariates"] = False
        super().__init__(**kw)
        self.c = nn.Parameter(torch.tensor(float(np.log(np.exp(0.5) - 1))))

    def alpha(self, tau_hat, m, s):
        return nn.functional.softplus(self.c) * torch.ones_like(tau_hat)


# Prior on the initial state: by 1 Dec 2020 about 1.6 % of the population had
# been reported as cases; allowing for under-reporting, we require the
# non-infected classes S+Q at tau=0 to lie in [0.85, 0.98], and V(0)=0
# (vaccination started on 14 Jan 2021).
SQ0_RANGE = (0.85, 0.98)
LAM_INIT = 100.0


def train_fpinn_init(data, cov, seed, use_cov=True, learn_z=True, z_fixed=None,
                     iters=(1500, 3000, 3000), lam_phys=100.0, lam_pop=100.0, lam_s=0.01,
                     w_d=1.0, lr=2e-3, lr_scalar=2e-2):
    """Same as fpinn_synthetic.train_fpinn, plus the initial-state prior."""
    torch.manual_seed(seed); np.random.seed(seed)
    z_init = float(np.random.uniform(0.75, 0.97))
    model = F.FPINN(use_covariates=use_cov, learn_z=learn_z, z_fixed=z_fixed, z_init=z_init)
    scal = [p for n, p in model.named_parameters() if n in ("zeta", "theta")]
    nets = [p for n, p in model.named_parameters() if n not in ("zeta", "theta")]
    opt = torch.optim.Adam([{"params": nets, "lr": lr}, {"params": scal, "lr": lr_scalar}])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=sum(iters) // 3, gamma=0.5)
    days = data["days_train"]
    y = torch.as_tensor(data["y_train"]); d = torch.as_tensor(data["d_train"])
    n1, n2, n3 = iters
    relu = nn.functional.relu
    for it in range(n1 + n2 + n3):
        lp = 0.0 if it < n1 else (lam_phys * min(1.0, (it - n1) / max(n2, 1)))
        q = F.forward_quantities(model, days, cov)
        yhat = model.varpi() * F.N_POP * (q["C"][1:] - q["C"][:-1])
        dhat = F.N_POP * (q["M"][1:] - q["M"][:-1])
        L_data = ((torch.log1p(torch.clamp(yhat, min=0)) - torch.log1p(y)) ** 2
                  + w_d * (torch.log1p(torch.clamp(dhat, min=0)) - torch.log1p(d)) ** 2).mean()
        L_phys = (q["resid"] ** 2).mean()
        L_pop = ((q["X"].sum(1) - 1.0) ** 2).mean()   # total population = 1 at every grid point
        L_s = ((q["a"][1:] - q["a"][:-1]) ** 2).mean()
        sq0 = q["X"][0, 0] + q["X"][0, 1]
        L_init = relu(SQ0_RANGE[0] - sq0) ** 2 + relu(sq0 - SQ0_RANGE[1]) ** 2 + q["X"][0, 7] ** 2
        loss = L_data + lp * L_phys + lam_pop * L_pop + lam_s * L_s + LAM_INIT * L_init
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        q = F.forward_quantities(model, days, cov)
        yhat = model.varpi() * F.N_POP * (q["C"][1:] - q["C"][:-1])
    return model, dict(z=float(q["z"].detach()), varpi=float(model.varpi().detach()),
                       alpha=q["a"].numpy(), X=q["X"].numpy(), C=q["C"].numpy(),
                       M=q["M"].numpy(), yhat=yhat.numpy(), L_data=float(L_data))


def re_curve(res, beta):
    """Effective reproduction number along the learned trajectory."""
    z = res["z"]; P = {k: v ** z for k, v in F.PARAMS.items()}
    X = res["X"]; a = res["alpha"]
    k1 = P["mu_A"] + P["phi_A"] + P["d"]
    k2 = P["mu_I"] + P["phi_I"] + P["rho_I"] + P["d"]
    S, V = X[:, 0], X[:, 7]
    return (a ** z) * P["psi"] * (S + P["sigma"] * V) / (P["psi"] + P["d"]) * (
        P["delta"] * (1 - P["eta"]) / k1 + P["eta"] / k2)


def nb_dispersion(y, mu):
    excess = np.mean((y - mu) ** 2 - mu)
    return float(np.mean(mu ** 2) / excess) if excess > 0 else 1e6


def evaluate(runs, y_test, rng, n_draw=50):
    Fc = np.array([r["forecast"] for r in runs])               # (K, H)
    disp = np.median([r["disp"] for r in runs])
    draws = []
    for f in Fc:
        mu = np.maximum(f, 1e-6)
        p = disp / (disp + mu)
        draws.append(rng.negative_binomial(disp, p, size=(n_draw, len(mu))))
    D = np.concatenate(draws).astype(float)                    # predictive samples
    med = np.median(Fc, axis=0)
    lo, hi = np.quantile(D, 0.025, axis=0), np.quantile(D, 0.975, axis=0)
    zs = np.array([r["z"] for r in runs]); ws = np.array([r["varpi"] for r in runs])
    return dict(z_med=float(np.median(zs)), z_lo=float(np.quantile(zs, .025)),
                z_hi=float(np.quantile(zs, .975)), varpi_med=float(np.median(ws)),
                varpi_lo=float(np.quantile(ws, .025)), varpi_hi=float(np.quantile(ws, .975)),
                mae=float(np.mean(np.abs(med - y_test))),
                rmse=float(np.sqrt(np.mean((med - y_test) ** 2))),
                coverage=float(np.mean((y_test >= lo) & (y_test <= hi))),
                wis=F.wis(y_test, D), L_data=float(np.median([r["L_data"] for r in runs])),
                disp=float(disp), med=med, lo=lo, hi=hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data_tr"); ap.add_argument("--out", default="results_tr")
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--K", type=int, default=None)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    K = args.K or (2 if args.quick else 10)
    iters = (150, 300, 300) if args.quick else (1500, 3000, 3000)

    D = load(args.data)
    T_TOTAL = len(D["dates"]) - 1; T_TRAIN = T_TOTAL - H_TEST
    F.T_TOTAL, F.T_TRAIN, F.N_POP = T_TOTAL, T_TRAIN, D["pop"]
    days = np.arange(T_TOTAL + 1, dtype=float)
    beta = D["doses"] / D["pop"]
    F.mobility = lambda t: np.interp(t, days, D["m"])
    F.stringency = lambda t: np.interp(t, days, D["s"])
    F.beta_fun = lambda t: np.interp(t, days, beta)
    # daily counts y_j, j = 1..T_TOTAL, are the reports of day j (date index j)
    data = dict(days=np.arange(T_TOTAL + 1), y=D["cases"][1:], d=D["deaths"][1:])
    data = F.prepare(data)
    cov = F.cov_on(data["days_train"].astype(float))
    y_test = data["y"][T_TRAIN:]
    rng = np.random.default_rng(0)

    configs = [("M1", "fPINN (learned $z$, covariates)", dict(), None),
               ("M2", "PINN ($z=1$, covariates)", dict(learn_z=False, z_fixed=0.9999), None),
               ("M3", "Fractional, constant $\\alpha$", dict(use_cov=False), ConstAlphaFPINN)]
    out, store = {}, {}
    orig = F.FPINN
    for key, label, kw, cls in configs:
        print(key, label, flush=True)
        F.FPINN = cls or orig
        runs = []
        for k in range(K):
            t0 = time.time()
            model, res = train_fpinn_init(data, cov, seed=2000 + k, iters=iters, **kw)
            res["forecast"] = F.forecast(model, res, data, None)
            res["disp"] = nb_dispersion(data["y_train"], np.maximum(res["yhat"], 1e-6))
            res["Re"] = re_curve(res, beta)
            runs.append(res)
            print(f"  seed {k}: z={res['z']:.3f} varpi={res['varpi']:.3f} "
                  f"L_data={res['L_data']:.4f} {time.time()-t0:.0f}s", flush=True)
        F.FPINN = orig
        store[key] = runs
        out[key] = evaluate(runs, y_test, rng)
        out[key]["label"] = label

    # ---------------- tables ----------------
    with open(os.path.join(args.out, "table_turkey.tex"), "w") as f:
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n"
                "Model & $\\hat z$ & $\\hat\\varpi$ & MAE & RMSE & 95\\% coverage & WIS \\\\\n\\midrule\n")
        for key, _, _, _ in configs:
            e = out[key]
            zt = (f"{e['z_med']:.3f} ({e['z_lo']:.3f}, {e['z_hi']:.3f})" if key != "M2" else "1 (fixed)")
            f.write(f"{e['label']} & {zt} & {e['varpi_med']:.2f} & {e['mae']:.0f} & {e['rmse']:.0f} & "
                    f"{100*e['coverage']:.0f}\\% & {e['wis']:.0f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---------------- figures ----------------
    dates = D["dates"]; dtr = dates[1:T_TRAIN + 1]; dte = dates[T_TRAIN + 1:]
    lock = (pd.Timestamp("2021-04-29"), pd.Timestamp("2021-05-17"))
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, key in zip(axs, ["M1", "M3"]):
        runs, e = store[key], out[key]
        fit = np.median([r["yhat"] for r in runs], axis=0)
        ax.axvspan(*lock, color="0.9", label="full lockdown")
        ax.plot(dates[1:], data["y"], ".", ms=3, color="0.4", label="reported (7-day avg)")
        ax.plot(dtr, fit, color="C0", label="fit")
        ax.plot(dte, e["med"], color="C3", label="forecast")
        ax.fill_between(dte, e["lo"], e["hi"], color="C3", alpha=.25, lw=0)
        ax.axvline(dates[T_TRAIN], color="k", ls=":", lw=.8)
        ax.set_title(e["label"].replace("$", "").replace("\\alpha", "α").replace("z", "z"))
        ax.tick_params(axis="x", rotation=30)
    axs[0].set_ylabel("daily reported cases"); axs[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "fig_turkey_cases.pdf")); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    A = np.array([r["alpha"] for r in store["M1"]]); R = np.array([r["Re"] for r in store["M1"]])
    dtr0 = dates[:T_TRAIN + 1]
    for ax, Y, lab in [(axs[0], A, "transmission rate α(τ)"), (axs[1], R, "effective reproduction number")]:
        ax.axvspan(*lock, color="0.9")
        ax.fill_between(dtr0, np.quantile(Y, .025, 0), np.quantile(Y, .975, 0), color="C0", alpha=.25, lw=0)
        ax.plot(dtr0, np.median(Y, 0), color="C0", label="median")
        ax.set_ylabel(lab); ax.tick_params(axis="x", rotation=30)
    axs[1].axhline(1, color="k", lw=.8, ls="--")
    ax2 = axs[0].twinx(); ax2.plot(dtr0, D["s"][:T_TRAIN + 1], color="C1", lw=1, label="stringency")
    ax2.plot(dtr0, D["m"][:T_TRAIN + 1], color="C2", lw=1, label="mobility")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("covariate (scaled)"); ax2.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "fig_turkey_alpha_re.pdf")); plt.close(fig)

    # ---------------- summary ----------------
    Re_med = np.median(R, 0)
    below = dtr0[Re_med < 1]
    with open(os.path.join(args.out, "summary_turkey.txt"), "w") as f:
        f.write(f"Period {START} to {END}; train {dates[0].date()} - {dates[T_TRAIN].date()} "
                f"({T_TRAIN} daily values), test {dte[0].date()} - {dte[-1].date()} ({H_TEST} days)\n")
        f.write(f"Population {D['pop']:.0f}; 10 Dec 2020 raw report {D['raw_dump']:.0f} replaced\n")
        f.write(f"Cases (7-day avg): max {np.nanmax(data['y']):.0f}; deaths max {np.nanmax(data['d']):.0f}\n")
        f.write(f"K = {K}, iterations = {iters}; rho_I = {F.PARAMS['rho_I']}, rho_D = {F.PARAMS['rho_D']}\n")
        for key, _, _, _ in configs:
            e = out[key]
            f.write(f"{key} {e['label']}: z {e['z_med']:.3f} [{e['z_lo']:.3f}, {e['z_hi']:.3f}], "
                    f"varpi {e['varpi_med']:.3f} [{e['varpi_lo']:.3f}, {e['varpi_hi']:.3f}], "
                    f"MAE {e['mae']:.0f}, RMSE {e['rmse']:.0f}, coverage {100*e['coverage']:.0f}%, "
                    f"WIS {e['wis']:.0f}, data loss {e['L_data']:.4f}, NB dispersion {e['disp']:.0f}\n")
        f.write(f"M1 alpha: median at start {np.median(A[:,0]):.3f}, min {np.median(A,0).min():.3f} on "
                f"{dtr0[np.argmin(np.median(A,0))].date()}, max {np.median(A,0).max():.3f} on "
                f"{dtr0[np.argmax(np.median(A,0))].date()}, at end {np.median(A[:,-1]):.3f}\n")
        f.write(f"M1 Re: max {Re_med.max():.2f} on {dtr0[np.argmax(Re_med)].date()}, "
                f"min {Re_med.min():.2f} on {dtr0[np.argmin(Re_med)].date()}; "
                f"days with Re<1: {len(below)}" +
                (f" (first {below[0].date()}, last {below[-1].date()})" if len(below) else "") + "\n")
        X0 = np.median([r["X"][0] for r in store["M1"]], 0); XT = np.median([r["X"][-1] for r in store["M1"]], 0)
        f.write("M1 state at start: " + ", ".join(f"{n}={x:.4f}" for n, x in zip(F.NAMES, X0)) + "\n")
        f.write("M1 state at end of training: " + ", ".join(f"{n}={x:.4f}" for n, x in zip(F.NAMES, XT)) + "\n")
        f.write(f"M1 test y: observed mean {y_test.mean():.0f}, forecast median mean {out['M1']['med'].mean():.0f}\n")
    pd.DataFrame({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)}
                  for k, v in out.items()}).T.to_csv(os.path.join(args.out, "results_turkey.csv"))
    with open(os.path.join(args.out, "settings_turkey.json"), "w") as f:
        json.dump(dict(START=START, END=END, H_TEST=H_TEST, K=K, iters=iters, PARAMS=F.PARAMS,
                       N_POP=D["pop"], lam_phys=100.0, lam_s=0.01, lam_pop_all_times=100.0,
                       smoothing="trailing 7-day mean", SQ0_RANGE=SQ0_RANGE, LAM_INIT=LAM_INIT, dump_fix="2020-12-10 replaced by neighbour mean"),
                  f, indent=2)
    print("Done", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
