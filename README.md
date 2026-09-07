# NHL Predictor Pipeline

Predicts NHL game outcomes and projects full-season standings for **2026-27**. Pairs
**MoneyPuck team analytics** (expected goals, Corsi, high-danger shots) with **game results
from the free NHL API**, trains a soft-voting ensemble (Logistic Regression + Gradient
Boosting), and converts per-game win probabilities into projected records by division,
conference and league.

Honest headline: **the pipeline works end to end, but the model is only slightly better than
guessing** — +1.5pp over the home-ice baseline, about one standard error. That number is the
interesting part of this project, not something to hide.

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

MoneyPuck files are **not** downloaded automatically. Grab each season-summary `teams.csv`
from [moneypuck.com/data.htm](https://moneypuck.com/data.htm) and save as
`data/raw/teams_<season>.csv`, where `<season>` is the year the season *started* (2025-26 →
`teams_2025.csv`). You need **one more team file than you have game seasons**, since each
season's games are matched to the *previous* season's profiles: for games 2022-2025 →
`teams_2021.csv` through `teams_2025.csv`.

```bash
python -m src.fetch_games        # ~3 min, rate-limited -> games_<season>.csv (labels)
python -m src.build_dataset      # -> data/processed/dataset.csv
python -m src.train              # -> models/ensemble.joblib
python -m src.predict_standings  # -> headline, standings, findings, next steps
```

The 2026-27 schedule caches to `data/raw/schedule_2026.csv`; delete it to re-fetch.

## How it works

**The leakage rule.** To predict a game in season N, only use team strength from season N-1.
A season's own end-of-year stats already "saw" the games you're predicting, so season-N games
are always paired with season-(N-1) profiles.

**Features.** Each row is one game, described by the *gap* between the two teams:
`diff_<stat> = home_prior_season_stat - away_prior_season_stat`. Nine total — three percentages
(xGoals%, Corsi%, Fenwick%) and six counting stats as per-game rates.

**Franchise continuity.** Arizona became Utah on 2024-04-18. Both source files store whatever
code was used that season, so `ARI` and `UTA` each appear on both sides depending on year.
`canonical_team()` is applied to **both** join keys at lookup time rather than baked into either
file, leaving `data/raw/` a faithful record of each source.

## Results

Trained on 2022-2024 (3,936 games), tested on the held-out 2025-26 season (1,312 games).

| Model | Accuracy | ROC-AUC | Log loss | Brier |
|---|---|---|---|---|
| Home-ice baseline | 0.5221 | — | — | — |
| Logistic regression | 0.5335 | **0.5509** | **0.6902** | **0.2485** |
| Gradient boosting | 0.5282 | 0.5256 | 0.6963 | 0.2515 |
| **Ensemble** | **0.5373** | 0.5458 | 0.6913 | 0.2491 |

Every metric has a "no-skill" reference — knowing it is the whole game. **Accuracy** must beat
the home-ice baseline (0.5221: pick the home team every game, using zero information); **ROC-AUC**
must beat 0.500 (coin flip; measures ranking, not calibration); **log loss** must beat 0.6931
(`ln 2`, saying "50%" every game); **Brier** must beat 0.2500.

**The honest read.** The ensemble beats the baseline by +1.5pp, but at n=1,312 one standard error is
±1.38pp — barely one s.e., not something to bet on. Log loss and Brier squeak just under their
references, so the probabilities carry a little real signal (AUC ≈ 0.55). Two quirks: **gradient
boosting is the weaker member** (worse than flat-0.5 on log loss and Brier; it overfits nine
features — try `weights=[3, 1]` or `max_depth=2`), and **`home_ice` is a dead feature** (constant `1`,
so `StandardScaler` zeroes it and trees can't split on it; home advantage is absorbed by the
intercept). Why so weak? Prior-season team strength is thin evidence for one hockey game — goalie
form, injuries, rest and luck swamp season-long quality. Richer public models top out near 57-60%;
53.7% is the honest ceiling of these nine features.

### Home-ice baseline by season

The bar any model must clear — and it moves, which is why it is recomputed per test season rather
than hard-coded. A window containing a high-home-win year raises the bar, making the model look
worse with no change to the model itself.

| Season | 2022-23 | 2023-24 | 2024-25 | 2025-26 | pooled |
|---|---|---|---|---|---|
| Home win % | 52.4% | 54.1% | **56.2%** | 52.2% | 53.7% |

## On train/test splits

Two alternatives were tested. **Neither is an improvement.**

| Split | Train n | Baseline | Accuracy | Edge | AUC |
|---|---|---|---|---|---|
| **Current: train 2022-24 / test 2025** | 3,936 | 0.5221 | 0.5373 | **+1.52pp** | 0.5458 |
| Train 2021-23 / test 2024-25 | 2,624 | 0.5423 | 0.5469 | +0.46pp | 0.5640 |
| Overlapping: train 2022-24 / test 2023-25 | 3,936 | 0.5419 | 0.5727 | +3.07pp | 0.5936 |

**Season 2021 has zero game rows** — the dataset covers game seasons 2022-2025, and `teams_2021.csv`
exists only as a *feature* source. So "train 2021-2023" is really just 2022+2023: it silently drops a
third of the training data and the edge collapses to +0.46pp, which at ±0.98pp s.e. is
indistinguishable from zero. **The overlapping split is invalid** — 2023 and 2024 sit in both sides,
and its +3.07pp is fake: on seasons it trained on it scores 0.5903, on the truly unseen 2025 only
0.5373. **A higher baseline is not a better model** — it is just the home-win rate of whichever
seasons land in the test set (table above), so any window containing 2024-25 mechanically lifts it,
making the job *harder*.

**The Florida case.** FLA is the most depressed team league-wide, xGoals% falling from the 91st
percentile (2024) to the 41st (2025). But the split cannot fix this: the projection reads its features
from `teams_2025.csv` regardless of which seasons the model *trains* on — across all three configs FLA
lands at 93.8 / 95.5 / 96.3 points (rank 16 / 13 / 12). The split isn't the lever; the features are.
So the features were tested.

## Multi-season blending: tested and rejected

The obvious fix is to blend 2-3 prior seasons into each profile so one injury-wrecked year can't
define a team. **It was tested and it makes the model worse, so it was not adopted — the code
intentionally still uses prior-season-only features.** Variants were evaluated on **identical game
rows** with walk-forward folds, pooled to n=2,624 (1 s.e. = ±0.98pp):

| Feature variant | Accuracy | Edge | AUC | Log loss |
|---|---|---|---|---|
| **Prior season only (current)** | **0.5526** | **+1.03pp** | **0.5617** | **0.6841** |
| Blend 2yr (0.70/0.30) | 0.5495 | +0.72pp | 0.5552 | 0.6891 |
| Blend 2yr (0.60/0.40) | 0.5400 | −0.23pp | 0.5541 | 0.6902 |
| Blend 2yr (0.50/0.50) | 0.5434 | +0.11pp | 0.5524 | 0.6908 |
| Shrink toward league mean | 0.5526 | +1.03pp | 0.5618 | 0.6840 |

Degradation is **monotonic** — more weight on older seasons, worse on every metric. That dose-response
across four settings beats any single comparison, and it reproduces with logistic regression alone
(AUC 0.5613 → 0.5560 → 0.5538 → 0.5525), so it isn't an artifact of the weaker boosting member.

**Why it fails.** Older seasons are genuinely less predictive — rosters turn over 20-30% a year.
Predicting season N+1 xGoals% from season N gives **r = +0.720**; from N−1, +0.540; from N−2, +0.391.
Blending dilutes a strong signal with weak ones and **compresses the spread** the model needs (league
xGoals% std 0.0318 → 0.0289 → 0.0283). A narrower test (2024-25 only, n=1,312) *did* favour a 2-year
blend, but with half the test data and a third of the training data it is noise.

**Correction to earlier advice:** shrinking toward the league mean was once listed here as a promising
lever. It is a mathematical **no-op** — shrinking is affine (`x' = wx + c`), `StandardScaler` removes
it exactly (max z-difference 5.3e-15), and the constant cancels in the home-minus-away difference.

### Regression and bounce-back candidates

Each team's 2025-26 profile against its own two-year baseline, flagging who is living above or below
their true level. Composite z across xGoals%, Corsi%, Fenwick%, goals and expected-goals for/against;
`Regressed est.` applies the measured lag-1 persistence `mean + 0.72 × (2025 − mean)`, league mean
xGoals% = 0.499.

| Team | xG% 2025 | 2yr base | Gap | Regressed est. | z | Signal |
|---|---|---|---|---|---|---|
| **ANA** | 0.510 | 0.445 | **+0.065** | 0.507 | **+1.78** | regression |
| SJS | 0.480 | 0.415 | +0.065 | 0.485 | +1.48 | regression |
| COL | 0.570 | 0.525 | +0.045 | 0.550 | +1.12 | regression |
| EDM | 0.520 | 0.560 | −0.040 | 0.514 | −1.07 | bounce-back |
| **FLA** | 0.500 | 0.560 | **−0.060** | 0.500 | **−1.40** | bounce-back |
| TOR | 0.460 | 0.505 | −0.045 | 0.471 | −1.49 | bounce-back |
| VAN | 0.450 | 0.505 | −0.055 | 0.464 | −1.55 | bounce-back |

**Anaheim is the single largest regression candidate in the league.** Read it carefully: the model
has **no offseason roster data whatsoever**. This flags *"ANA's 2025 was far above their
established level and is unlikely to repeat"* — it does **not** know Anaheim lost players. Those
claims happen to agree here. These tables are diagnostics, deliberately **not wired into the model**.

## Projected 2026-27 standings

`python -m src.predict_standings` leads with the headline call, then ranked tables by division,
conference and league, then findings and next steps. Writes
`data/processed/projected_standings_2026.csv`.

```
  PRESIDENTS' TROPHY FAVOURITE          DIVISION WINNERS
    COL   116.7 pts  (53.5-20.8-9.7)      Atlantic      TBL   104.7
                                          Metropolitan  CAR   108.8
  PROJECTED CELLAR                        Central       COL   116.7
    VAN    75.8 pts                       Pacific       VGK   102.5
```

Projected wins = the **sum of a team's per-game win probabilities** (home games contribute `p`, road
`1-p`), which yields the mean of the win distribution without a Monte Carlo loop. 2026-27 is an
**84-game** season under the new CBA, reflected automatically in the schedule.

- Projected spread is ~41 points best-to-worst vs. ~60 in a real season; a near-coin-flip model
  regresses everything toward .500, so treat the *ordering* as a weak signal.
- **OTL is not predicted** — a flat league constant (`config.OT_GAME_RATE = 0.23`), identical for
  every team, so it never changes the ordering.
- The saved model trains on 2022-2024 and **never sees 2025**, the season feeding the projection.
  Refitting on all four moves 21 of 32 teams (max 5 places, largest swing 4.8 pts; Spearman 0.98).
  Validate on a holdout, then refit on everything to forecast.

## Next steps

1. **Roster / player-level features** — biggest expected gain. The model knows only team aggregates
   and has no idea who is on the ice. MoneyPuck already publishes `skaters.csv`, `goalies.csv` and
   `lines.csv`; none are used. Goaltending is hockey's largest swing factor, and this is the only
   way to capture **offseason roster change**: team stats describe the team that *was*, not the one
   that *will be*.
2. **Injury / man-games-lost** — separates "collapsed from injuries" from "actually got worse", the
   missing half of the bounce-back question.
3. **Fix the ensemble weights** — gradient boosting scores worse than a flat 0.5 guess.
4. **Refit on all seasons before forecasting** (see caveat above).
5. **More training seasons — but not unlimited history.** More seasons means more training *rows*,
   which helps; it does not mean averaging old seasons into a profile (rejected above). Deep history
   is avoided deliberately: predictive power decays fast (r = 0.72 → 0.54 → 0.39 by lag); 2019-20
   and 2020-21 are structurally broken (COVID cut them to 56 games, and 2020-21 used realigned
   division-only schedules, so strength-of-schedule isn't comparable); expansion redrew the league
   twice (VGK 2017, SEA 2021); and rule changes mean 2015-era rates aren't the same game. Practical
   window: roughly 2021 onward, which is what this uses.

## Layout

```
src/config.py             paths, seasons, features, divisions   <- control panel
src/fetch_games.py        NHL API -> games_<season>.csv (labels)
src/build_dataset.py      leakage-safe feature/label join
src/train.py              ensemble + honest evaluation
src/predict_standings.py  headline call, standings, findings, next steps
data/raw/                 games_*.csv, teams_*.csv, schedule_2026.csv
data/processed/           dataset.csv, projected_standings_2026.csv
models/                   ensemble.joblib
```

`src/` has no `__init__.py` — it works as an implicit namespace package (PEP 420), fine for
`python -m src.<module>`. To package for distribution use `find_namespace_packages()`; setuptools'
`find_packages()` skips namespace packages.

**Sources:** [NHL API](https://api-web.nhle.com/v1) (results + schedules, no key; fields via
[Zmalski/NHL-API-Reference](https://github.com/Zmalski/NHL-API-Reference)) ·
[MoneyPuck](https://moneypuck.com/data.htm) (team summaries — **manual download only**; MoneyPuck
blocks scraping and asks that you contact them for a data licence, so `config.MONEYPUCK_URL` is a
reference, not a fetcher).

**Requires:** Python 3.9+, `pandas`, `numpy`, `scikit-learn`, `requests`, `joblib`.
