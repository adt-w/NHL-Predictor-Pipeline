"""
STEP 5 — Project every team's full-season record, grouped by division.

Loads the trained ensemble, applies it to the TARGET_SEASON schedule, and
turns per-game win probabilities into projected W / L / OTL / points.

Expected wins, not simulated wins: a team's projected win total is the SUM of
its per-game win probabilities. Summing probabilities gives the mean of the
win distribution directly, so no Monte Carlo loop is needed for the point
estimate. The tradeoff is that you get no spread (no "playoff odds"), and
because the model is only slightly better than a coin flip the projections
compress hard toward .500 — see the caveat printed at the end of the run.

Inputs:
    models/ensemble.joblib                (from src.train)
    data/raw/teams_<TARGET_SEASON-1>.csv  (prior-season profiles = the features)
    the TARGET_SEASON schedule            (NHL API, cached to data/raw/)

Run:  python -m src.predict_standings
"""
import time

import joblib
import pandas as pd
import requests

from src import config
from src.build_dataset import load_team_profiles, make_game_features

SCHEDULE_COLUMNS = ["game_id", "date", "season", "home", "away"]


def fetch_schedule(start_year: int) -> pd.DataFrame:
    """Return every scheduled regular-season game for one season.

    Unlike src.fetch_games (which keeps only COMPLETED games so it can build
    labels), this keeps games that have not been played yet — that is the
    whole point, since we are projecting a season before it happens.
    """
    season_code = config.nhl_season_code(start_year)
    rows = []

    for team in config.TEAMS:
        url = f"{config.NHL_API}/club-schedule-season/{team}/{season_code}"
        try:
            resp = requests.get(url, timeout=20)
        except requests.RequestException as e:
            print(f"[warn] {team} {season_code}: request failed ({e})")
            continue
        if resp.status_code != 200:
            print(f"[warn] {team} {season_code}: HTTP {resp.status_code}")
            continue
        time.sleep(0.4)

        for g in resp.json().get("games", []):
            if g.get("gameType") != 2:
                continue                      # regular season only
            rows.append({
                "game_id": g["id"],
                "date":    g["gameDate"],
                "season":  start_year,
                "home":    g["homeTeam"]["abbrev"],
                "away":    g["awayTeam"]["abbrev"],
            })

    df = pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)
    if df.empty:
        return df
    return df.drop_duplicates("game_id").reset_index(drop=True)


def load_schedule(start_year: int) -> pd.DataFrame:
    """Fetch the schedule once, then reuse the cached copy on later runs."""
    cached = config.RAW / f"schedule_{start_year}.csv"
    if cached.exists():
        return pd.read_csv(cached)

    df = fetch_schedule(start_year)
    if not df.empty:
        config.RAW.mkdir(parents=True, exist_ok=True)
        df.to_csv(cached, index=False)
        print(f"[ok] cached {len(df)} scheduled games -> {cached}")
    return df


def project_records(schedule: pd.DataFrame, prob_home: pd.Series) -> pd.DataFrame:
    """Turn per-game home-win probabilities into a projected record per team.

    Every game contributes p to the home team's expected wins and (1 - p) to
    the away team's, so each team's total is the sum over its own games.
    """
    df = schedule.assign(p=prob_home.values)

    wins = (df.groupby("home")["p"].sum()
              .add(df.groupby("away")["p"].apply(lambda s: (1 - s).sum()),
                   fill_value=0.0))
    played = (df["home"].value_counts()
                .add(df["away"].value_counts(), fill_value=0))

    out = pd.DataFrame({"GP": played, "W": wins}).fillna(0.0)
    out["L"] = out["GP"] - out["W"]

    # An OT/SO game hands the loser a point. Roughly OT_GAME_RATE of games get
    # there, and a team loses about half the ones it plays, so its OT losses
    # are ~ half that share of its schedule — capped by its actual losses.
    out["OTL"] = (out["GP"] * config.OT_GAME_RATE / 2).clip(upper=out["L"])
    out["REG_L"] = out["L"] - out["OTL"]
    out["PTS"] = 2 * out["W"] + out["OTL"]

    out["division"] = out.index.map(config.TEAM_DIVISION)
    out["conference"] = out.index.map(config.TEAM_CONFERENCE)
    if out["division"].isna().any():
        unknown = sorted(out.index[out["division"].isna()])
        raise KeyError(f"teams missing from config.DIVISIONS: {unknown}")
    return out


def _table(records: pd.DataFrame, title: str, indent: str = "") -> None:
    """Print one ranked standings block."""
    block = records.sort_values("PTS", ascending=False)
    print(f"\n{indent}{title}")
    print(f"{indent}{'#':>3}  {'TEAM':<5}{'GP':>4}{'W':>7}{'L':>7}"
          f"{'OTL':>6}{'PTS':>7}")
    print(f"{indent}{'-' * 41}")
    for rank, (team, r) in enumerate(block.iterrows(), start=1):
        print(f"{indent}{rank:>3}  {team:<5}{int(r['GP']):>4}{r['W']:>7.1f}"
              f"{r['REG_L']:>7.1f}{r['OTL']:>6.1f}{r['PTS']:>7.1f}")


def print_headline(records: pd.DataFrame, season: int) -> None:
    """The lede: who the model likes for 2026-27, before any methodology."""
    label = f"{season}-{str(season + 1)[-2:]}"
    bar = "=" * 45
    ranked = records.sort_values("PTS", ascending=False)
    top = ranked.iloc[0]

    print(f"\n{bar}")
    print(f"  NHL {label} — PROJECTED SEASON".center(45))
    print(bar)
    print(f"\n  PRESIDENTS' TROPHY FAVOURITE")
    print(f"    {top.name}   {top['PTS']:.1f} pts   "
          f"({top['W']:.1f}-{top['REG_L']:.1f}-{top['OTL']:.1f})")

    print(f"\n  DIVISION WINNERS")
    for conference, divisions in config.CONFERENCES.items():
        for division in divisions:
            w = records[records["division"] == division]["PTS"].idxmax()
            r = records.loc[w]
            print(f"    {division:<14}{w:<5}{r['PTS']:>7.1f} pts")

    print(f"\n  CONFERENCE LEADERS")
    for conference in config.CONFERENCES:
        w = records[records["conference"] == conference]["PTS"].idxmax()
        print(f"    {conference:<14}{w:<5}{records.loc[w, 'PTS']:>7.1f} pts")

    print(f"\n  PROJECTED CELLAR")
    bot = ranked.iloc[-1]
    print(f"    {bot.name}   {bot['PTS']:.1f} pts")


def print_standings(records: pd.DataFrame) -> None:
    bar = "=" * 45
    print(f"\n{bar}\nFULL STANDINGS\n{bar}")
    print(f"{len(records)} teams | expected wins summed from per-game win "
          f"probabilities")

    print(f"\n{bar}\nBY DIVISION\n{bar}")
    for conference, divisions in config.CONFERENCES.items():
        for division in divisions:
            _table(records[records["division"] == division],
                   f"{division} ({conference})", indent="  ")

    print(f"\n{bar}\nBY CONFERENCE\n{bar}")
    for conference in config.CONFERENCES:
        _table(records[records["conference"] == conference],
               f"{conference} Conference", indent="  ")

    print(f"\n{bar}\nLEAGUE-WIDE\n{bar}")
    _table(records, "All 32 teams", indent="  ")


def print_findings(records: pd.DataFrame) -> None:
    """Season-by-season home-ice baseline, the bar every model must clear."""
    bar = "=" * 45
    print(f"\n{bar}\nFINDINGS\n{bar}")

    print("\n  Home-ice baseline by season")
    print("  (share of games won by the home team — the accuracy you get")
    print("   by blindly picking the home side, using zero information)")
    print(f"\n  {'SEASON':<10}{'GAMES':>7}{'HOME WIN %':>13}")
    print(f"  {'-' * 30}")

    path = config.PROCESSED / "dataset.csv"
    if not path.exists():
        print("    dataset.csv not found — run `python -m src.build_dataset`")
        return

    df = pd.read_csv(path)
    by_season = df.groupby("season")["home_win"].agg(["size", "mean"])
    for s, r in by_season.iterrows():
        label = f"{s}-{str(s + 1)[-2:]}"
        print(f"  {label:<10}{int(r['size']):>7}{r['mean']:>12.1%}")
    print(f"  {'-' * 30}")
    print(f"  {'pooled':<10}{len(df):>7}{df['home_win'].mean():>12.1%}")

    hi, lo = by_season["mean"].idxmax(), by_season["mean"].idxmin()
    print(f"\n  Home advantage is NOT stable: it ranges from "
          f"{by_season.loc[lo, 'mean']:.1%} ({lo}-{str(lo+1)[-2:]}) to "
          f"{by_season.loc[hi, 'mean']:.1%} ({hi}-{str(hi+1)[-2:]}).")
    print("  This is why the baseline is recomputed per test season rather")
    print("  than hard-coded: a test window containing a high-home-win year")
    print("  raises the bar, which makes the model look worse without any")
    print("  change to the model itself.")

    spread = records["PTS"].max() - records["PTS"].min()
    print(f"\n  Projected point spread: {spread:.1f} (vs ~60 in a real season).")
    print("  A near-coin-flip model regresses every team toward .500, so read")
    print("  the ORDER as a weak signal and the point totals as estimates.")


def print_next_steps() -> None:
    bar = "=" * 45
    print(f"\n{bar}\nNEXT STEPS\n{bar}")

    print("""
  1. ROSTER / PLAYER-LEVEL FEATURES  (biggest expected gain)
     The model currently knows only team-aggregate rates. It has no idea
     who is on the ice. MoneyPuck already publishes skaters.csv,
     goalies.csv and lines.csv for every season — none are used yet.
       - Goaltending is the single largest swing factor in hockey; a
         starter's prior-season save percentage above expected is likely
         the highest-value feature not yet in the model.
       - Top-6 forward and top-4 defence quality, minutes-weighted.
       - This is also the only way to capture OFFSEASON ROSTER CHANGE.
         Team-level stats describe the team that WAS; a roster-aware
         model describes the team that WILL BE.

  2. INJURY / MAN-GAMES-LOST
     A team whose profile collapsed because of injuries looks identical
     to a team that simply got worse. Man-games-lost would separate the
     two and is the missing half of the bounce-back question.

  3. FIX THE ENSEMBLE WEIGHTS
     Gradient boosting currently scores worse than a flat 0.5 guess on
     log loss and Brier; logistic regression alone beats the ensemble on
     both. Try weights=[3, 1] or max_depth=2 in src/train.py.

  4. REFIT ON ALL SEASONS BEFORE FORECASTING
     The saved model trains on 2022-2024 and never sees 2025 — the very
     season whose stats feed this projection. Validate on a holdout,
     then refit on everything.

  5. MORE TRAINING SEASONS — BUT NOT UNLIMITED HISTORY
     More seasons means more training ROWS, which helps. It does NOT
     mean averaging old seasons into a team's profile: that was tested
     and rejected (see README). Deep history is deliberately avoided:
       - Predictive power decays fast. Season N predicts season N+1 at
         r = 0.72; two years back r = 0.54; three years back r = 0.39.
         Rosters turn over 20-30% a year.
       - 2019-20 and 2020-21 are structurally broken seasons: COVID cut
         them short (56 games) and 2020-21 used realigned, division-only
         schedules, so strength-of-schedule is not comparable.
       - Expansion redrew the league twice recently (VGK 2017,
         SEA 2021), and Arizona relocated to Utah in 2024.
       - Rule and officiating changes shift scoring environments, so
         2015-era rates are not measuring the same game.
     Practical window: roughly 2021 onward, which is what this uses.

  6. DATA ACCESS
     MoneyPuck blocks scraping and asks that you contact them for a data
     licence, so team files are downloaded by hand. A licence would let
     the pipeline fetch seasons automatically and widen the window.
""".rstrip())


def main() -> None:
    bundle_path = config.MODELS / "ensemble.joblib"
    if not bundle_path.exists():
        print(f"No model at {bundle_path} — run `python -m src.train` first.")
        return

    bundle = joblib.load(bundle_path)
    model, feature_cols = bundle["model"], bundle["features"]

    season = config.TARGET_SEASON
    schedule = load_schedule(season)
    if schedule.empty:
        print(f"No {season}-{str(season + 1)[-2:]} schedule published yet.")
        return

    prior = load_team_profiles(season - 1)
    feats = make_game_features(schedule, prior)
    if feats.empty:
        print("No scheduled games could be matched to prior-season profiles.")
        return

    # make_game_features may drop games, so realign the schedule to what
    # actually survived before attributing wins to teams.
    kept = schedule[schedule["game_id"].isin(feats["game_id"])].reset_index(drop=True)
    prob_home = pd.Series(model.predict_proba(feats[feature_cols])[:, 1])

    records = project_records(kept, prob_home)

    print_headline(records, season)
    print_standings(records)
    print_findings(records)
    print_next_steps()

    out_path = config.PROCESSED / f"projected_standings_{season}.csv"
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    records.sort_values("PTS", ascending=False).to_csv(out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
