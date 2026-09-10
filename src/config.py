"""
Central configuration. Edit the constants here instead of hard-coding paths
in every script. Nothing in this file needs heavy logic — it's your control panel.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

# ---------------------------------------------------------------------------
# Seasons
# MoneyPuck labels a season by the YEAR IT STARTS. So "2025" == the 2025-26 season.
# VERIFY this on moneypuck.com before trusting it (see PLAN.md, Step 2 checkpoint).
#
# You need at least TWO seasons so you can train leakage-safe:
#   features come from season N, labels come from season N+1 games.
# Download more seasons (they're free) to grow the training set.
# ---------------------------------------------------------------------------
SEASONS = [2022, 2023, 2024, 2025]        # MoneyPuck season IDs you have/will download

# The season whose games you'll predict once its schedule is released.
TARGET_SEASON = 2026                       # 2026-27

# NHL API uses an 8-digit season code: 2025-26 -> "20252026"
def nhl_season_code(start_year: int) -> str:
    return f"{start_year}{start_year + 1}"

# ---------------------------------------------------------------------------
# Data source URL patterns  (VERIFY these live — this environment is offline)
# ---------------------------------------------------------------------------
# MoneyPuck season-summary CSVs. Confirm the exact path by right-clicking a
# download link on https://moneypuck.com/data.htm
MONEYPUCK_URL = (
    "https://moneypuck.com/moneypuck/playerData/seasonSummary/"
    "{season}/regular/{table}.csv"      # table in {teams, skaters, goalies, lines}
)

# NHL public JSON API base. A team's full-season schedule (with final scores) is at:
#   {NHL_API}/club-schedule-season/{TEAM}/{SEASONCODE}
# Cross-check field names against the community reference:
#   https://github.com/Zmalski/NHL-API-Reference
NHL_API = "https://api-web.nhle.com/v1"

# All 32 team abbreviations (matches the 'team' column in your CSVs)
TEAMS = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET",
    "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT",
    "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK",
    "WPG", "WSH",
]

# ---------------------------------------------------------------------------
# Divisional alignment, used to group projected standings.
# Utah inherited Arizona's Central Division slot for 2024-25.
# ---------------------------------------------------------------------------
CONFERENCES = {"Eastern": ["Atlantic", "Metropolitan"],
               "Western": ["Central", "Pacific"]}

DIVISIONS = {
    "Atlantic":     ["BOS", "BUF", "DET", "FLA", "MTL", "OTT", "TBL", "TOR"],
    "Metropolitan": ["CAR", "CBJ", "NJD", "NYI", "NYR", "PHI", "PIT", "WSH"],
    "Central":      ["CHI", "COL", "DAL", "MIN", "NSH", "STL", "UTA", "WPG"],
    "Pacific":      ["ANA", "CGY", "EDM", "LAK", "SEA", "SJS", "VAN", "VGK"],
}

TEAM_DIVISION = {t: d for d, ts in DIVISIONS.items() for t in ts}
TEAM_CONFERENCE = {t: c for c, ds in CONFERENCES.items()
                   for d in ds for t in DIVISIONS[d]}

# Share of NHL games that reach overtime/shootout. The loser of one of those
# games still banks a point (an "OTL"), so this splits projected losses into
# regulation losses vs. OT losses for the points column. League-average
# constant, NOT something the model predicts per-team.
OT_GAME_RATE = 0.23

# ---------------------------------------------------------------------------
# Feature columns pulled from teams.csv (situation == "all", position "Team Level").
# Start small and defensible. Add more once the baseline works.
# These are RATE-friendly or already percentages; you'll divide counts by
# games_played in build_dataset.py to make them per-game.
# ---------------------------------------------------------------------------
TEAM_PCT_FEATURES = [
    "xGoalsPercentage",
    "corsiPercentage",
    "fenwickPercentage",
]

# Counting stats you'll convert to per-game rates (value / games_played)
TEAM_COUNT_FEATURES = [
    "goalsFor",
    "goalsAgainst",
    "xGoalsFor",
    "xGoalsAgainst",
    "highDangerShotsFor",
    "highDangerShotsAgainst",
]
