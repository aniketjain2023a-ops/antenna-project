# Data folder

| File | What it is |
|---|---|
| `synthetic_demo.csv` | **SYNTHETIC** demonstration dataset (500 rows). Created automatically by `datasets/synthetic.py` on first start if missing. |
| `metaopt_ai.db` | SQLite database (created automatically; not committed to git). |
| `registered/` | Cleaned copies of datasets uploaded through the dashboard (not committed). |

## The synthetic dataset is NOT real antenna data

`synthetic_demo.csv` is produced from a documented, physics-*inspired* analytical
formula plus seeded Gaussian noise. It is **not** an electromagnetic (CST/HFSS/
openEMS) simulation and **not** a VNA or chamber measurement. Every row carries
`data_label = SYNTHETIC`, and the dataset is registered in SQLite with
`dataset_type = SYNTHETIC` and a full provenance string.

Columns:

| Column | Unit | Meaning |
|---|---|---|
| `sample_id` | - | 1..500 |
| `L_mm` | mm | effective patch/metasurface length, Latin-hypercube sampled in [8.0, 12.5] |
| `G_mm` | mm | metasurface gap, sampled in [0.2, 2.0] |
| `S11_dB` | dB | reflection coefficient at the assumed 10 GHz design frequency |
| `Gain_dBi` | dBi | realized broadside gain |
| `data_label` | - | always `SYNTHETIC` |

### Generating equations (all assumptions)

```
dL(G)      = 0.4 / (G + 0.4)                                    [mm]  gap loading
f_r(L,G)   = c / (2 (L + 2 dL) sqrt(eps_eff)),  eps_eff = 2.0          half-wave resonance
delta      = (f_r - F0) / F0,                   F0 = 10 GHz
Gamma_min  = 0.05 + 0.25 (1 - exp(-((G - 0.8)/0.6)^2))                 best match near G = 0.8 mm
|Gamma|^2  = 1 - (1 - Gamma_min^2) / (1 + (2 Q delta)^2),  Q = 5        single-resonance model
S11_dB     = 10 log10 |Gamma|^2  + N(0, 0.6 dB),  clipped to <= -0.05 dB (passivity)
D_dBi      = 6.5 + 0.35 (L - 8) + 1.2 exp(-((G - 1.3)/0.45)^2)          metasurface gain term peaks at G = 1.3 mm
Gain_dBi   = D_dBi + 10 log10(1 - |Gamma|^2) - 0.3 dB + N(0, 0.15 dB)   mismatch + fixed loss
```

Seeds: Latin hypercube `seed=42`, noise `default_rng(43)`. Regenerating with the
same seed and sample count reproduces the file bit-for-bit (SHA-256 is stored in
the database and in every report).

Bandwidth is **not** generated. The application never invents a bandwidth
column; it is only used if a real dataset supplies `Bandwidth_MHz`.

## Using real data

Upload CSV/Excel in **Dataset Management**, map columns, choose the dataset type
(`SIMULATION`, `MEASUREMENT` or `PUBLIC`), write the provenance (solver/instrument,
settings, date, source) and confirm the column meanings. The original file is
never modified; a cleaned copy is stored in `registered/`.
