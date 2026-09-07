# NHL Predictor Pipeline

Predicts NHL game outcomes and projects full-season standings for **2026-27**.

Pairs **MoneyPuck team analytics** (expected goals, Corsi, high-danger shots) with
**game results from the free NHL API**, trains a soft-voting ensemble
(Logistic Regression + Gradient Boosting), and converts per-game win probabilities into
projected records by division, conference, and league.

Honest headline: **the pipeline works end to end, but the model is only slightly better
than guessing** — +1.5pp over the home-ice baseline, which is about one standard error.
That number is the interesting part of this project, not something to hide.

---

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

MoneyPuck files are **not** downloaded automatically. Grab each season-summary
`teams.csv` from [moneypuck.com/data.htm](https://moneypuck.com/data.htm) and save as
`data/raw/teams_<season>.csv`, where `<season>` is the year the season *started*
(2025-26 → `teams_2025.csv`).

You need **one more team file than you have game seasons**, since each season's games are
matched to the *previous* season's profiles. For games 2022-2025 → `teams_2021.csv`
through `teams_2025.csv`.

```bash
python -m src.fetch_games        # ~3 min, rate-limited -> games_<season>.csv (labels)
python -m src.build_dataset      # -> data/processed/dataset.csv
python -m src.train              # -> models/ensemble.joblib
python -m src.predict_standings  # -> projected standings (division/conference/league)
```

The 2026-27 schedule caches to `data/raw/schedule_2026.csv`; delete it to re-fetch.

---

## How it works

| Script | Role |
|---|---|
| `fetch_games.py` | NHL API → completed games. These are the **labels** (`home_win`). |
| `build_dataset.py` | Joins team profiles to game results, leakage-safe. |
| `train.py` | Trains + evaluates the ensemble. |
| `predict_standings.py` | Applies the model to the 2026-27 schedule. |

**The leakage rule.** To predict a game in season N, only use team strength from season
N-1. A season's own end-of-year stats already "saw" the games you're predicting, so
season-N games are always paired with season-(N-1) profiles.

**Features.** Each row is one game, described by the *gap* between the two teams:
`diff_<stat> = home_prior_season_stat - away_prior_season_stat`. Nine total — three
percentages (xGoals%, Corsi%, Fenwick%) and six counting stats as per-game rates.

**Franchise continuity.** Arizona became Utah on 2024-04-18. Both source files store
whatever code was used that season, so `ARI` and `UTA` each appear on both sides
depending on year. `canonical_team()` is applied to **both** join keys at lookup time
rather than baked into either file, leaving `data/raw/` a faithful record of each source.

---

## Results

Trained on 2022-2024 (3,936 games), tested on the held-out 2025-26 season (1,312 games).

| Model | Accuracy | ROC-AUC | Log loss | Brier |
|---|---|---|---|---|
| Home-ice baseline | 0.5221 | — | — | — |
| Logistic regression | 0.5335 | **0.5509** | **0.6902** | **0.2485** |
| Gradient boosting | 0.5282 | 0.5256 | 0.6963 | 0.2515 |
| **Ensemble** | **0.5373** | 0.5458 | 0.6913 | 0.2491 |

Every metric has a "no-skill" reference — knowing it is the whole game:

- **Accuracy** → beat the **home-ice baseline** (0.5221): predict the home team every
  game. That's the floor; it uses zero information.
- **ROC-AUC** → beat **0.500** (coin flip). Measures ranking, not calibration.
- **Log loss** → beat **0.6931** (`ln 2`, i.e. saying "50%" every game).
- **Brier** → beat **0.2500** (same flat-0.5 reference).

**The honest read.** The ensemble beats the baseline by +1.5pp, but at n=1,312 one
standard error is ±1.38pp — the edge is barely one s.e. and not something to bet on. Log
loss and Brier squeak just under their references, so the probabilities carry a little
real signal, consistent with AUC ≈ 0.55.

Two things worth knowing:

1. **Gradient boosting is the weaker member** — worse than flat-0.5 on log loss (0.6963)
   and Brier (0.2515); it overfits nine features. Logistic regression alone has the best
   AUC, log loss, and Brier here. Try `weights=[3, 1]` or `max_depth=2`.
2. **`home_ice` is a dead feature** — constant `1`, so `StandardScaler` zeroes it and
   trees can't split on it. Home advantage is absorbed by the model intercept.

Why so weak? Prior-season team strength is thin evidence for a single hockey game. Goalie
form, injuries, rest, and luck swamp season-long quality. Richer public models top out
near 57-60%; 53.7% is the honest ceiling of these nine features.

---

## On train/test splits

Two alternatives were tested against the current split. **Neither is an improvement.**

| Split | Train n | Baseline | Accuracy | Edge | AUC |
|---|---|---|---|---|---|
| **Current: train 2022-24 / test 2025** | 3,936 | 0.5221 | 0.5373 | **+1.52pp** | 0.5458 |
| Train 2021-23 / test 2024-25 | 2,624 | 0.5423 | 0.5469 | +0.46pp | 0.5640 |
| Overlapping: train 2022-24 / test 2023-25 | 3,936 | 0.5419 | 0.5727 | +3.07pp | 0.5936 |

**Season 2021 has zero game rows.** The dataset covers game seasons 2022-2025;
`teams_2021.csv` exists only as a *feature* source (the prior-season profile for 2022
games). So "train 2021-2023" is really just 2022+2023 — it silently drops a third of the
training data (3,936 → 2,624) and the edge over baseline collapses to +0.46pp, which at
±0.98pp s.e. is statistically indistinguishable from zero.

**The overlapping split is invalid** — 2023 and 2024 sit in both sides. Its +3.07pp is
entirely fake: on seasons it trained on it scores 0.5903, on the truly unseen 2025 it
scores 0.5373.

**A higher baseline is not a better model.** The home-ice baseline is just the home-win
rate of whichever seasons land in the test set — a property of the **data**:

| Season | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| Home-win rate | 0.5236 | 0.5412 | **0.5625** | 0.5221 |

Any test window containing 2024 mechanically lifts the baseline, which raises the bar the
model must clear. It makes the job *harder*, not the model better.

The real improvement is **more history** — MoneyPuck team files back to 2015, then extend
`config.SEASONS`. More seasons beat any reshuffling of four.

### On the "Florida bounce-back" case

FLA is the most depressed team league-wide — xGoals% from the 91st percentile (2024) to
the **41st** (2025). But **the split cannot fix this**: the 2026-27 projection reads its
features from `teams_2025.csv` regardless of which seasons the model *trains* on. Across
all three training configs FLA lands at 93.8 / 95.5 / 96.3 points (rank 16 / 13 / 12) — a
2.5-point range. The split isn't the lever; the features are. So the features were tested.

---

## Multi-season blending: tested and rejected

The obvious fix for the FLA case is to blend 2-3 prior seasons into each team profile, so
one injury-wrecked year can't define a team. **It was tested and it makes the model worse,
so it was not adopted — the code intentionally still uses prior-season-only features.**

Every variant was evaluated on **identical game rows** with walk-forward folds (train on
past, test on future), predictions pooled to n=2,624 (1 s.e. = ±0.98pp):

| Feature variant | Accuracy | Edge | AUC | Log loss |
|---|---|---|---|---|
| **Prior season only (current)** | **0.5526** | **+1.03pp** | **0.5617** | **0.6841** |
| Blend 2yr (0.70/0.30) | 0.5495 | +0.72pp | 0.5552 | 0.6891 |
| Blend 2yr (0.60/0.40) | 0.5400 | −0.23pp | 0.5541 | 0.6902 |
| Blend 2yr (0.50/0.50) | 0.5434 | +0.11pp | 0.5524 | 0.6908 |
| Shrink toward league mean | 0.5526 | +1.03pp | 0.5618 | 0.6840 |

The degradation is **monotonic** — the more weight on older seasons, the worse every
metric gets. That dose-response across four settings is far more convincing than any
single comparison, and it reproduces with logistic regression alone (AUC 0.5613 → 0.5560 →
0.5538 → 0.5525), so it isn't an artifact of the weaker gradient-boosting member.

**Why it fails.** Older seasons are genuinely less predictive of next season — NHL rosters
turn over 20-30% a year:

| Predicting season N+1 xGoals% from... | Correlation |
|---|---|
| season N (lag 1) | **r = +0.720** |
| season N−1 (lag 2) | r = +0.540 |
| season N−2 (lag 3) | r = +0.391 |

Blending dilutes a strong signal (0.72) with weaker ones. It also **compresses the spread**
the model depends on: league xGoals% std falls 0.0318 → 0.0289 (2yr) → 0.0283 (3yr),
squeezing teams toward the middle of an already near-coin-flip model.

One caveat on the evidence: a narrower test (game seasons 2024-25 only, n=1,312, trained on
a single season) *did* favour a 2-year blend. It's contradicted by the better-powered test
above, has half the test data and a third of the training data, so it's treated as noise.

**Correction to earlier advice in this README:** shrinking team strength toward the league
mean was previously listed as a promising lever. It is mathematically a **no-op** here.
Shrinking is an affine transform (`x' = wx + c`), `StandardScaler` removes it exactly
(verified: max z-score difference 5.3e-15), and because the constant cancels in the
home-minus-away difference it only rescales the features. Logistic regression is invariant
to that by construction. The identical row in the table above is not a coincidence.

**What would actually help** (untested, needs data we don't have): injury / man-games-lost
features, and offseason roster-change signals. Neither exists in the current sources.

### Regression and bounce-back candidates

A useful by-product: comparing each team's 2025-26 profile against its own two-year
baseline flags who is likely living above or below their true level. Composite z-score
across xGoals%, Corsi%, Fenwick%, goals and expected-goals for/against.

**Prone to regression** (2025 far above own baseline):

| Team | xG% 2025 | 2yr baseline | Gap | Regressed est. | z |
|---|---|---|---|---|---|
| **ANA** | 0.510 | 0.445 | **+0.065** | 0.507 | **+1.78** |
| SJS | 0.480 | 0.415 | +0.065 | 0.485 | +1.48 |
| COL | 0.570 | 0.525 | +0.045 | 0.550 | +1.12 |
| MTL | 0.500 | 0.455 | +0.045 | 0.500 | +1.11 |
| OTT | 0.550 | 0.505 | +0.045 | 0.536 | +1.04 |

**Bounce-back candidates** (2025 far below own baseline):

| Team | xG% 2025 | 2yr baseline | Gap | Regressed est. | z |
|---|---|---|---|---|---|
| VAN | 0.450 | 0.505 | −0.055 | 0.464 | −1.55 |
| TOR | 0.460 | 0.505 | −0.045 | 0.471 | −1.49 |
| **FLA** | 0.500 | 0.560 | **−0.060** | 0.500 | **−1.40** |
| EDM | 0.520 | 0.560 | −0.040 | 0.514 | −1.07 |
| WPG | 0.480 | 0.515 | −0.035 | 0.485 | −1.05 |

`Regressed est.` applies the measured lag-1 persistence: `mean + 0.72 × (2025 − mean)`,
league mean xGoals% = 0.499.

**Anaheim is the single largest regression candidate in the league** — its 2025 xGoals%
(0.510) sits far above its own two-year baseline (0.445). Read this carefully though: the
model has **no offseason roster data whatsoever**. This flags *"ANA's 2025 was far above
their established level and is unlikely to be repeated"* — it does **not** know Anaheim
lost players. Those are different claims that happen to point the same direction here.
These tables are diagnostics, deliberately **not wired into the model**.

---

## Projected 2026-27 standings

`python -m src.predict_standings` leads with the **headline call** (Presidents' Trophy
favourite, division winners, conference leaders), then ranked tables **by division, by
conference, and league-wide**, then **findings** and **next steps**. It writes
`data/processed/projected_standings_2026.csv`.

```
  PRESIDENTS' TROPHY FAVOURITE          DIVISION WINNERS
    COL   116.7 pts  (53.5-20.8-9.7)      Atlantic      TBL   104.7
                                          Metropolitan  CAR   108.8
  PROJECTED CELLAR                        Central       COL   116.7
    VAN    75.8 pts                       Pacific       VGK   102.5
```

**Home-ice baseline by season** — the bar any model must clear, and it moves:

| Season | 2022-23 | 2023-24 | 2024-25 | 2025-26 | pooled |
|---|---|---|---|---|---|
| Home win % | 52.4% | 54.1% | **56.2%** | 52.2% | 53.7% |

It is recomputed per test season rather than hard-coded, because a test window containing
a high-home-win year raises the bar and makes the model look worse with no change to the
model itself.

Projected wins = the **sum of a team's per-game win probabilities** (home games contribute
`p`, road games `1-p`). Summing probabilities yields the mean of the win distribution, so
no Monte Carlo loop is needed for a point estimate. 2026-27 is an **84-game** season under
the new CBA, which the schedule data reflects automatically.

Read with care:

- Projected spread is ~41 points best-to-worst vs. ~60 in a real season. A near-coin-flip
  model regresses everything toward .500 — treat the *ordering* as a weak signal.
- **OTL is not predicted.** It's a flat league constant (`config.OT_GAME_RATE = 0.23`),
  identical for every team, so it never changes the ordering.
- The saved model trains on 2022-2024 and **never sees 2025** — the season whose stats
  feed the projection. Refitting on all four moves 21 of 32 teams (max 5 places, largest
  swing 4.8 pts; Spearman 0.98). Standard practice: validate on a holdout, then refit on
  everything before forecasting.

---

## Layout

```
src/
  config.py             paths, seasons, features, divisions   <- control panel
  fetch_games.py        NHL API -> games_<season>.csv (labels)
  build_dataset.py      leakage-safe feature/label join
  train.py              ensemble + honest evaluation
  predict_standings.py  headline call, standings, findings, next steps
data/raw/               games_*.csv, teams_*.csv, schedule_2026.csv
data/processed/         dataset.csv, projected_standings_2026.csv
models/                 ensemble.joblib
```

`src/` has no `__init__.py` — it works as an implicit namespace package (PEP 420), which
is fine for `python -m src.<module>`. If you ever package this for distribution, use
`find_namespace_packages()`, since setuptools' `find_packages()` skips namespace packages.

**Sources:** [NHL API](https://api-web.nhle.com/v1) (results + schedules, no key;
fields via [Zmalski/NHL-API-Reference](https://github.com/Zmalski/NHL-API-Reference)) ·
[MoneyPuck](https://moneypuck.com/data.htm) (team summaries — **manual download only**;
MoneyPuck blocks programmatic scraping and asks that you contact them for a data licence,
so `config.MONEYPUCK_URL` is a reference, not a fetcher).

**Requires:** Python 3.9+, `pandas`, `numpy`, `scikit-learn`, `requests`, `joblib`.
