"""
STEP 3 — Join FEATURES to LABELS into one model-ready table. Turn team profiles + game results into a model-ready table.

Inputs:
    data/raw/games_<season>.csv     (from src.fetch_games)
    data/raw/teams_<season>.csv     (from MoneyPuck, see src.fetch_moneypuck)

Leakage rule: features come from season N, labels come from season N+1 games.
A team's season-N stats must never be built from the games you are predicting.

The leakage rule (READ THIS):
    To predict a game in season N, only use team strength from season N-1.
    Using the SAME season's end-of-year stats to predict that season's games
    leaks the answer (the stats already "saw" those games). We avoid it by
    pairing season N-1 profiles with season N games.

Output: data/processed/dataset.csv  — one row per game, with diff features + label.

Run:  python -m src.build_dataset
"""

import pandas as pd

from src import config


# ---------------------------------------------------------------------------
# Franchise continuity.
# The Arizona Coyotes relocated and became the Utah Hockey Club on 2024-04-18.
# The NHL API uses "ARI" through 2023-24 and "UTA" from 2024-25 onward, with no
# overlap. The roster and hockey operations carried over intact, so Arizona's
# prior-season team stats are the legitimate feature row for Utah's next season.
#
# Both games_<season>.csv and teams_<season>.csv keep whatever code the
# franchise actually used that season, so "ARI" and "UTA" each show up on
# BOTH the games side and the profiles side depending on the year (only the
# one season that straddles the move needs translating). canonical_team is
# therefore applied to both join keys at lookup time in make_game_features,
# not baked into either source. Leave data/raw/ untouched — it stays a
# faithful record of what each source actually returned.
# ---------------------------------------------------------------------------
FRANCHISE_ALIASES = {
    "ARI": "UTA",   # Arizona Coyotes -> Utah (2024-25 onward)
    "PHX": "UTA",   # Phoenix Coyotes -> Arizona -> Utah (pre-2014 data only)
    "ATL": "WPG",   # Atlanta Thrashers -> Winnipeg Jets (2011)
}


def canonical_team(abbrev: str) -> str:
    """Map a historical team abbreviation onto its present-day franchise code."""
    return FRANCHISE_ALIASES.get(abbrev, abbrev)


def load_team_profiles(season: int) -> pd.DataFrame:
    """Load one season's teams.csv and return a clean per-team feature row.

    Filters to the season-total, even-strength-inclusive line:
        situation == "all"   (and the team-level row)
    Then builds per-game rates from counting stats.
    Index the result by team abbreviation so lookups are easy.
    """
    path = config.RAW / f"teams_{season}.csv"
    df = pd.read_csv(path)

    # MoneyPuck's teams.csv has a duplicated 'team' header column; pandas
    # renames the second occurrence to 'team.1'. We only ever reference
    # 'team' (the first), so 'team.1' is simply ignored below.
    df = df[df["situation"] == "all"].copy()

    feature_cols = list(config.TEAM_PCT_FEATURES) + list(config.TEAM_COUNT_FEATURES)
    required = ["team", "games_played"] + feature_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"teams_{season}.csv is missing expected columns: {missing}")

    # Guard against any stray duplicate team rows, then index by team code.
    df = df.drop_duplicates(subset="team", keep="first").set_index("team")

    features = df[config.TEAM_PCT_FEATURES].astype(float).copy()
    for col in config.TEAM_COUNT_FEATURES:
        features[f"{col}_pg"] = df[col] / df["games_played"]

    features.index.name = "team"
    return features


def make_game_features(games: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    """Join each game to the PRIOR-season profiles of its home & away teams,
    then create difference features (home minus away) + a home-ice constant.

    Difference features work well here: the model cares about the GAP in
    strength between the two teams, not their absolute levels.
    """
    # Normalize BOTH sides of the join through canonical_team. A relocation
    # like ARI->UTA can land on either side depending on the season: games
    # and profiles from before the move both say "ARI", games and profiles
    # from after both say "UTA", and only the one pairing that straddles the
    # move (season-N games in the new code vs. season-N-1 profiles in the
    # old code) actually needs translating. Mapping both sides through the
    # same alias table handles all three cases with one rule instead of
    # guessing which side is stale.
    games = games.copy()
    games["home"] = games["home"].map(canonical_team)
    games["away"] = games["away"].map(canonical_team)
    prior = prior.rename(index=canonical_team)

    known = games["home"].isin(prior.index) & games["away"].isin(prior.index)
    dropped = int((~known).sum())
    if dropped:
        print(f"[info] dropping {dropped} game(s) with a team missing from prior-season profiles")
    games = games.loc[known].reset_index(drop=True)

    # Row-aligned lookups: home_feats.iloc[i] / away_feats.iloc[i] are the
    # prior-season profiles for games.iloc[i]'s home/away teams.
    home_feats = prior.loc[games["home"]].reset_index(drop=True)
    away_feats = prior.loc[games["away"]].reset_index(drop=True)

    diffs = home_feats.subtract(away_feats)
    diffs.columns = [f"diff_{c}" for c in diffs.columns]

    out = pd.concat([games[["game_id", "season"]], diffs], axis=1)
    out["home_ice"] = 1
    # Unplayed games (a future schedule) carry no label — the same feature
    # code then serves both training and prediction.
    if "home_win" in games.columns:
        out["home_win"] = games["home_win"]
    return out


def main() -> None:
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    frames = []

    # Pair season N games with season N-1 profiles.
    for season in config.SEASONS:
        prior_season = season - 1
        try:
            games = pd.read_csv(config.RAW / f"games_{season}.csv")
            prior = load_team_profiles(prior_season)
            frames.append(make_game_features(games, prior))
        except (FileNotFoundError, NotImplementedError, KeyError) as e:
            print(f"[skip] {season}: {type(e).__name__}: {e}")
            continue

    if not frames:
        print("Nothing built yet — implement the TODOs and download the seasons.")
        return

    dataset = pd.concat(frames, ignore_index=True)
    out = config.PROCESSED / "dataset.csv"
    dataset.to_csv(out, index=False)
    print(f"[ok] wrote {len(dataset)} game-rows, "
          f"{dataset['season'].nunique()} seasons -> {out}")


if __name__ == "__main__":
    main()