"""
STEP 3 — Join FEATURES to LABELS into one model-ready table.

Inputs:
    data/raw/games_<season>.csv     (from src.fetch_games)
    data/raw/teams_<season>.csv     (from MoneyPuck, see src.fetch_moneypuck)

Leakage rule: features come from season N, labels come from season N+1 games.
A team's season-N stats must never be built from the games you are predicting.

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
# Apply this to the FEATURE side only, so season-N "ARI" stats line up with
# season-N+1 "UTA" games. Leave data/raw/ untouched — it stays a faithful
# record of what each source actually returned.
# ---------------------------------------------------------------------------
FRANCHISE_ALIASES = {
    "ARI": "UTA",   # Arizona Coyotes -> Utah (2024-25 onward)
    "PHX": "UTA",   # Phoenix Coyotes -> Arizona -> Utah (pre-2014 data only)
    "ATL": "WPG",   # Atlanta Thrashers -> Winnipeg Jets (2011)
}


def canonical_team(abbrev: str) -> str:
    """Map a historical team abbreviation onto its present-day franchise code."""
    return FRANCHISE_ALIASES.get(abbrev, abbrev)
