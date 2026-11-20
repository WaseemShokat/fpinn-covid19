"""Profile-loss experiment for the memory order z (identifiability, Section 6.1).

For synthetic data with z* = 0.8 and 0.9 (low noise) we
  (a) fix z on a grid and train K networks for each value, recording the data loss,
      the learned reporting fraction, the mean level of the learned transmission rate
      and the forecast error;
  (b) train K networks with learned z and record (z, varpi, mean alpha) for each member,
      to quantify the trade-off between the three.
Usage: python profile_z.py --out ../res_profile [--quick]
"""
import argparse, json, os, time
import numpy as np
import fpinn_synthetic as F

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="res_profile")
ap.add_argument("--quick", action="store_true")
ap.add_argument("--K", type=int, default=4)
ap.add_argument("--K_learn", type=int, default=10)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
iters = (150, 300, 300) if a.quick else (1500, 3000, 3000)
z_grid = [0.6, 0.7, 0.8, 0.9, 0.9999]
rows = []


def record(tag, zt, zfix, k, model, res, data):
    a_true = F.alpha_true(data["days_train"].astype(float), data["a0"])
    y_test = data["y"][F.T_TRAIN:]
    fc = F.forecast(model, res, data, None)
    rows.append(dict(set=tag, z_true=zt, z_fixed=zfix, seed=k, z=res["z"], varpi=res["varpi"],
                     alpha_mean=float(np.mean(res["alpha"])),
                     alpha_err=F.rel_l2(res["alpha"], a_true),
                     L_data=res["L_data"], test_mae=float(np.mean(np.abs(fc - y_test))),
                     min_forecast=float(np.min(fc))))
    print(tag, zt, zfix, k, {kk: round(v, 4) for kk, v in rows[-1].items() if isinstance(v, float)},
          flush=True)


for zt in (0.8, 0.9):
    data = F.prepare(F.generate_data(zt, 100.0, seed=int(zt * 100) + 100))   # same data as E1, low noise
    cov = F.cov_on(data["days_train"])
    for zf in z_grid:
        for k in range(a.K):
            model, res = F.train_fpinn(data, cov, seed=1000 + k, iters=iters,
                                       learn_z=False, z_fixed=zf)
            record("fixed", zt, zf, k, model, res, data)
    for k in range(a.K_learn):
        model, res = F.train_fpinn(data, cov, seed=1000 + k, iters=iters)
        record("learned", zt, None, k, model, res, data)
    json.dump(rows, open(os.path.join(a.out, "profile_rows.json"), "w"), indent=1)

json.dump(rows, open(os.path.join(a.out, "profile_rows.json"), "w"), indent=1)
print("done")
