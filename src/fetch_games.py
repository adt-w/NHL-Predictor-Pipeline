"""
STEP 1 — Get game RESULTS (your training labels).

Goal: produce one CSV per season at data/raw/games_<season>.csv with columns:
    game_id, date, season, home, away, home_goals, away_goals, home_win

Your MoneyPuck CSVs do NOT contain game logs, so we pull results from the
free NHL JSON API. No key required.

Run:  python -m src.fetch_games
"""
import time
import requests
import pandas as pd

from src import config

GAME_COLUMNS = [
    "game_id", "date", "season", "home", "away",
    "home_goals", "away_goals", "home_win",
]


def fetch_season_games(start_year: int) -> pd.DataFrame:
    """Return a tidy DataFrame of completed regular-season games for one season.

    Approach: loop the 32 teams, hit the club-schedule-season endpoint, and
    collect each game once (dedupe on game_id at the end).

    Endpoint: {NHL_API}/club-schedule-season/{TEAM}/{SEASONCODE}
    Each game carries: id, gameDate, gameType, gameState,
    homeTeam{abbrev, score}, awayTeam{abbrev, score}.
    gameType 1 = preseason, 2 = regular season, 3 = playoffs.
    gameState "OFF"/"FINAL" = played; "FUT" = scheduled, no score key yet.
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
        payload = resp.json()
        time.sleep(0.4)

        for g in payload.get("games", []):
            if g.get("gameType") != 2:
                continue                      # drop preseason (1) and playoffs (3)
            if g.get("gameState") not in ("OFF", "FINAL"):
                continue                      # drop FUT (not played yet)

            rows.append({
                "game_id":    g["id"],
                "date":       g["gameDate"],
                "season":     start_year,
                "home":       g["homeTeam"]["abbrev"],
                "away":       g["awayTeam"]["abbrev"],
                "home_goals": g["homeTeam"]["score"],
                "away_goals": g["awayTeam"]["score"],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=GAME_COLUMNS)
    df = df.drop_duplicates("game_id").reset_index(drop=True)
    df["home_win"] = (df["home_goals"] > df["away_goals"]).astype(int)
    return df


def main() -> None:
    config.RAW.mkdir(parents=True, exist_ok=True)
    for season in config.SEASONS + [config.TARGET_SEASON]:
        df = fetch_season_games(season)
        out = config.RAW / f"games_{season}.csv"
        df.to_csv(out, index=False)
        if df.empty:
            print(f"[warn] {season}: 0 completed games (season not played yet?) -> {out}")
        else:
            print(f"[ok] {season}: wrote {len(df)} games -> {out}")


if __name__ == "__main__":
    main()
