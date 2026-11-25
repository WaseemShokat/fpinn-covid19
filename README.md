# Fractional PINNs with time-varying transmission for COVID-19

Code for the paper

> W. Shokat, "Fractional Physics-Informed Neural Networks with Time-Varying Transmission for
> Modelling and Scenario Forecasting of COVID-19 Dynamics", submitted to *Chaos, Solitons & Fractals* (2026).

**What it does.** A fractional (Caputo) compartmental COVID-19 model is combined with a
physics-informed neural network. The network learns the memory order *z*, a time-varying
transmission rate α(t) (from time, mobility and policy stringency) and the reporting fraction
from daily cases and deaths. The L1 scheme for the Caputo derivative is built into the loss.

**Main findings.**
- Synthetic experiments with a known memory order and transmission rate test what the method can recover.
- The memory order *z* is only weakly identifiable: the data loss is almost flat in *z*,
  while forecasts depend strongly on it (see `profile_z.py`).
- Real-data applications: Türkiye (Dec 2020 – Jun 2021) and Pakistan (Jan – Jul 2021).

## Files

| File | What it does | Paper section |
|---|---|---|
| `fpinn_synthetic.py` | Model, L1 solver, fPINN, synthetic experiments and ablations | 6.1 |
| `fpinn_turkey.py` | Real-data experiment for Türkiye, 1 Dec 2020 – 30 Jun 2021 | 6.2 |
| `fpinn_pakistan.py` | Real-data experiment for Pakistan, 1 Jan – 31 Jul 2021 | 6.3 |
| `figures/fig_model.tex` | TikZ source of the model flow diagram | 3.1 |
| `figures/fig_framework.tex` | TikZ source of the fPINN framework diagram | 4.1 |

## Requirements

Python 3.10+:

```
pip install -r requirements.txt
```

## Data (all public)

Put these files in one folder, for example `data/`:

- `jhu_cases.csv`, `jhu_deaths.csv` – JHU CSSE time series
  (`time_series_covid19_confirmed_global.csv`, `time_series_covid19_deaths_global.csv`) from
  https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series
- `vacc.csv` – Our World in Data vaccinations:
  https://github.com/owid/covid-19-data/blob/master/public/data/vaccinations/vaccinations.csv
- `oxcgrt.csv` – OxCGRT compact national file:
  https://github.com/OxCGRT/covid-policy-dataset (`data/OxCGRT_compact_national_v1.csv`)
- `2020_TR_Region_Mobility_Report.csv`, `2021_TR_Region_Mobility_Report.csv`,
  `2020_PK_Region_Mobility_Report.csv`, `2021_PK_Region_Mobility_Report.csv` – from
  https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip

## Running

```
python fpinn_synthetic.py --out results              # about 1.5–2 h on a CPU (--quick for a test)
python fpinn_turkey.py   --data data --out results_tr
python fpinn_pakistan.py --data data --out results_pk
```

Each run writes tables (`.tex`), figures (`.pdf`), a `summary*.txt` with all numbers used
in the paper, and a `settings*.json` file with the settings.

## Non-negative observation model and identifiability (Section 6.5)

By default the daily cases and deaths are differences of the fractional integrals C and M
(the main model of the paper). To use the non-negative flow model of Section 6.5, in which
the daily cases are the flow into the diagnosed class, set the environment variable
`FPINN_OBS=flow`:

```
FPINN_OBS=flow python fpinn_synthetic.py --out results_flow
FPINN_OBS=flow python fpinn_turkey.py   --data data --out results_tr_flow
FPINN_OBS=flow python fpinn_pakistan.py --data data --out results_pk_flow
FPINN_OBS=flow python profile_z.py      --out results_profile   # profile of the data loss in z (Figure 9)
```

`profile_z.py` trains networks with the memory order fixed at 0.6, 0.7, 0.8, 0.9 and 1, and
with learned order, on the synthetic data with true order 0.8 and 0.9, and saves all results
to `profile_rows.json`.

## Citation

If you use this code, please cite the paper above (see also `CITATION.cff`).

## Contact

Waseem Shokat – waseemshokat91@gmail.com – ORCID [0009-0007-0398-0146](https://orcid.org/0009-0007-0398-0146)
