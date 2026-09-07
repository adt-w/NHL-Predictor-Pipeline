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


# def load_schedule(start_year: int) -> pd.DataFrame:
#     """Fetch the schedule once, then reuse the cached copy on later runs."""
#     cached = config.RAW / f"schedule_{start_year}.csv"
#     if cached.exists():
#         return pd.read_csv(cached)

#     df = fetch_schedule(start_year)
#     if not df.empty:
#         config.RAW.mkdir(parents=True, exist_ok=True)
#         df.to_csv(cached, index=False)
#         print(f"[ok] cached {len(df)} scheduled games -> {cached}")
#     return df


# def project_records(schedule: pd.DataFrame, prob_home: pd.Series) -> pd.DataFrame:
#     """Turn per-game home-win probabilities into a projected record per team.

#     Every game contributes p to the home team's expected wins and (1 - p) to
#     the away team's, so each team's total is the sum over its own games.
#     """
#     df = schedule.assign(p=prob_home.values)

#     wins = (df.groupby("home")["p"].sum()
#               .add(df.groupby("away")["p"].apply(lambda s: (1 - s).sum()),
#                    fill_value=0.0))
#     played = (df["home"].value_counts()
#                 .add(df["away"].value_counts(), fill_value=0))

#     out = pd.DataFrame({"GP": played, "W": wins}).fillna(0.0)
#     out["L"] = out["GP"] - out["W"]

#     # An OT/SO game hands the loser a point. Roughly OT_GAME_RATE of games get
#     # there, and a team loses about half the ones it plays, so its OT losses
#     # are ~ half that share of its schedule — capped by its actual losses.
#     out["OTL"] = (out["GP"] * config.OT_GAME_RATE / 2).clip(upper=out["L"])
#     out["REG_L"] = out["L"] - out["OTL"]
#     out["PTS"] = 2 * out["W"] + out["OTL"]

#     out["division"] = out.index.map(config.TEAM_DIVISION)
#     if out["division"].isna().any():
#         unknown = sorted(out.index[out["division"].isna()])
#         raise KeyError(f"teams missing from config.DIVISIONS: {unknown}")
#     return out


# def print_standings(records: pd.DataFrame, season: int) -> None:
#     label = f"{season}-{str(season + 1)[-2:]}"
#     print(f"\nProjected {label} standings "
#           f"(expected wins from {len(records)} teams' game-by-game probabilities)")

#     for conference, divisions in config.CONFERENCES.items():
#         print(f"\n{'=' * 52}\n{conference.upper()} CONFERENCE\n{'=' * 52}")
#         for division in divisions:
#             block = (records[records["division"] == division]
#                      .sort_values("PTS", ascending=False))
#             print(f"\n  {division}")
#             print(f"  {'Team':<6}{'GP':>4}{'W':>7}{'L':>7}{'OTL':>7}{'PTS':>7}")
#             for team, r in block.iterrows():
#                 print(f"  {team:<6}{int(r['GP']):>4}{r['W']:>7.1f}"
#                       f"{r['REG_L']:>7.1f}{r['OTL']:>7.1f}{r['PTS']:>7.1f}")


# def main() -> None:
#     bundle_path = config.MODELS / "ensemble.joblib"
#     if not bundle_path.exists():
#         print(f"No model at {bundle_path} — run `python -m src.train` first.")
#         return

#     bundle = joblib.load(bundle_path)
#     model, feature_cols = bundle["model"], bundle["features"]

#     season = config.TARGET_SEASON
#     schedule = load_schedule(season)
#     if schedule.empty:
#         print(f"No {season}-{str(season + 1)[-2:]} schedule published yet.")
#         return

#     prior = load_team_profiles(season - 1)
#     feats = make_game_features(schedule, prior)
#     if feats.empty:
#         print("No scheduled games could be matched to prior-season profiles.")
#         return

#     # make_game_features may drop games, so realign the schedule to what
#     # actually survived before attributing wins to teams.
#     kept = schedule[schedule["game_id"].isin(feats["game_id"])].reset_index(drop=True)
#     prob_home = pd.Series(model.predict_proba(feats[feature_cols])[:, 1])

#     records = project_records(kept, prob_home)
#     print_standings(records, season)

#     out_path = config.PROCESSED / f"projected_standings_{season}.csv"
#     config.PROCESSED.mkdir(parents=True, exist_ok=True)
#     records.sort_values("PTS", ascending=False).to_csv(out_path)
#     print(f"\nSaved -> {out_path}")

#     spread = records["PTS"].max() - records["PTS"].min()
#     print(f"\nCaveat: projected point spread is only {spread:.1f}. The model "
#           f"barely beats the home-ice baseline, so these standings are far\n"
#           f"flatter than a real NHL season (~60 points between best and worst). "
#           f"Treat the ORDER as a weak signal and the point totals as\n"
#           f"regression-to-the-mean estimates, not forecasts.")


if __name__ == "__main__":
    main()
