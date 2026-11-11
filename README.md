# Fractional PINNs with time-varying transmission for COVID-19

Code for the paper "Fractional Physics-Informed Neural Networks with Time-Varying
Transmission for Modelling and Forecasting COVID-19 Dynamics" (W. Shokat).

## Files

| File | What it does | Paper section |
|---|---|---|
| `fpinn_synthetic.py` | Model, L1 solver, fPINN, synthetic experiments and ablations | 6.1 |
| `fpinn_turkey.py` | Real-data experiment for Türkiye, 1 Dec 2020 – 30 Jun 2021 | 6.2 |
| `fpinn_pakistan.py` | Real-data experiment for Pakistan, 1 Jan – 31 Jul 2021 | 6.3 |
| `figures/fig_model.tex` | TikZ source of the model flow diagram | 3.1 |
| `figures/fig_framework.tex` | TikZ source of the fPINN framework diagram | 4.1 |

## Requirements

Python 3.10+ with `torch`, `numpy`, `pandas`, `matplotlib`:

```
pip install torch numpy pandas matplotlib
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
