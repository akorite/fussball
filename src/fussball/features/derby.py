"""Derby and rivalry features.

Identifies city derbies, regional rivalries, and computes a historical
intensity metric based on average total goals in recent head-to-head meetings.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from fussball.data.schema import MatchRecord

# ---------------------------------------------------------------------------
# City mapping: team -> city
# ---------------------------------------------------------------------------

_CITY_MAP: dict[str, str] = {
    # Manchester
    "Manchester United": "Manchester",
    "Manchester City": "Manchester",
    "Man United": "Manchester",
    "Man Utd": "Manchester",
    "Man City": "Manchester",
    # London
    "Arsenal": "London",
    "Tottenham": "London",
    "Tottenham Hotspur": "London",
    "Chelsea": "London",
    "West Ham": "London",
    "West Ham United": "London",
    "Crystal Palace": "London",
    "Fulham": "London",
    # Liverpool
    "Liverpool": "Liverpool",
    "Everton": "Liverpool",
    # Milan
    "AC Milan": "Milan",
    "Inter Milan": "Milan",
    "Inter": "Milan",
    # Madrid
    "Real Madrid": "Madrid",
    "Atletico Madrid": "Madrid",
    "Atletico": "Madrid",
    # Barcelona
    "Barcelona": "Barcelona",
    "Espanyol": "Barcelona",
    # Turin
    "Juventus": "Turin",
    "Torino": "Turin",
    # Glasgow
    "Celtic": "Glasgow",
    "Rangers": "Glasgow",
    # Dortmund
    "Borussia Dortmund": "Dortmund",
    "Schalke": "Gelsenkirchen",
    "Schalke 04": "Gelsenkirchen",
}

# ---------------------------------------------------------------------------
# Country mapping: team -> country
# ---------------------------------------------------------------------------

_COUNTRY_MAP: dict[str, str] = {
    # England (E0)
    "Arsenal": "England",
    "Aston Villa": "England",
    "Bournemouth": "England",
    "Brentford": "England",
    "Brighton": "England",
    "Burnley": "England",
    "Chelsea": "England",
    "Crystal Palace": "England",
    "Everton": "England",
    "Fulham": "England",
    "Ipswich": "England",
    "Leicester": "England",
    "Leeds": "England",
    "Liverpool": "England",
    "Luton": "England",
    "Manchester City": "England",
    "Manchester United": "England",
    "Man City": "England",
    "Man United": "England",
    "Man Utd": "England",
    "Newcastle": "England",
    "Nottm Forest": "England",
    "Nottingham Forest": "England",
    "Sheffield": "England",
    "Sheffield United": "England",
    "Southampton": "England",
    "Tottenham": "England",
    "Tottenham Hotspur": "England",
    "West Brom": "England",
    "West Ham": "England",
    "West Ham United": "England",
    "Wolves": "England",
    # Spain (SP1)
    "Real Madrid": "Spain",
    "Barcelona": "Spain",
    "Atletico Madrid": "Spain",
    "Atletico": "Spain",
    "Sevilla": "Spain",
    "Real Sociedad": "Spain",
    "Real Betis": "Spain",
    "Villarreal": "Spain",
    "Athletic Bilbao": "Spain",
    "Celta Vigo": "Spain",
    "Espanyol": "Spain",
    "Getafe": "Spain",
    "Mallorca": "Spain",
    "Osasuna": "Spain",
    "Rayo Vallecano": "Spain",
    "Valencia": "Spain",
    "Alaves": "Spain",
    "Girona": "Spain",
    "Las Palmas": "Spain",
    "Leganes": "Spain",
    "Real Valladolid": "Spain",
    "Eibar": "Spain",
    "Huesca": "Spain",
    "Cadiz": "Spain",
    "Elche": "Spain",
    "Granada": "Spain",
    "Levante": "Spain",
    "Almeria": "Spain",
    # Germany (D1)
    "Bayern Munich": "Germany",
    "Borussia Dortmund": "Germany",
    "Bayer Leverkusen": "Germany",
    "RB Leipzig": "Germany",
    "Eintracht Frankfurt": "Germany",
    "Wolfsburg": "Germany",
    "Borussia Monchengladbach": "Germany",
    "Gladbach": "Germany",
    "Freiburg": "Germany",
    "Union Berlin": "Germany",
    "Hoffenheim": "Germany",
    "Stuttgart": "Germany",
    "Werder Bremen": "Germany",
    "Mainz": "Germany",
    "Augsburg": "Germany",
    "Hertha Berlin": "Germany",
    "Darmstadt": "Germany",
    "Heidenheim": "Germany",
    "Dusseldorf": "Germany",
    "Schalke": "Germany",
    "Schalke 04": "Germany",
    "Koln": "Germany",
    "Nurnberg": "Germany",
    "Hamburg": "Germany",
    # Italy (I1)
    "Inter Milan": "Italy",
    "Inter": "Italy",
    "AC Milan": "Italy",
    "Juventus": "Italy",
    "Napoli": "Italy",
    "Roma": "Italy",
    "Atalanta": "Italy",
    "Lazio": "Italy",
    "Fiorentina": "Italy",
    "Torino": "Italy",
    "Bologna": "Italy",
    "Sassuolo": "Italy",
    "Udinese": "Italy",
    "Sampdoria": "Italy",
    "Genoa": "Italy",
    "Cagliari": "Italy",
    "Lecce": "Italy",
    "Empoli": "Italy",
    "Frosinone": "Italy",
    "Monza": "Italy",
    "Verona": "Italy",
    "Salernitana": "Italy",
    "Spezia": "Italy",
    "Cremonese": "Italy",
    "Venezia": "Italy",
    "Parma": "Italy",
    "Como": "Italy",
    "Hellas Verona": "Italy",
    # France (F1)
    "Paris Saint-Germain": "France",
    "PSG": "France",
    "Marseille": "France",
    "Lyon": "France",
    "Monaco": "France",
    "Lille": "France",
    "Nice": "France",
    "Rennes": "France",
    "Lens": "France",
    "Strasbourg": "France",
    "Nantes": "France",
    "Montpellier": "France",
    "Brest": "France",
    "Reims": "France",
    "Toulouse": "France",
    "Le Havre": "France",
    "Lorient": "France",
    "Clermont": "France",
    "Metz": "France",
    "Auxerre": "France",
    "Saint-Etienne": "France",
    "Angers": "France",
    # Netherlands (N1)
    "Ajax": "Netherlands",
    "Feyenoord": "Netherlands",
    "PSV": "Netherlands",
    "AZ": "Netherlands",
    "FC Utrecht": "Netherlands",
    "Twente": "Netherlands",
    "Vitesse": "Netherlands",
    "Heerenveen": "Netherlands",
    "Groningen": "Netherlands",
}

# ---------------------------------------------------------------------------
# Historical intensity window
# ---------------------------------------------------------------------------

_INTENSITY_WINDOW = 5
"""Number of past meetings to average for historical intensity."""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_derby_features(records: list[MatchRecord]) -> pd.DataFrame:
    """Compute derby and rivalry features.

    Returns DataFrame with match_id as index and columns:

    - ``same_city``: 1 if both teams are from the same city (city derby)
    - ``same_region``: 1 if both teams are from the same country
    - ``historical_intensity``: avg total goals in last 5 meetings between the
      two teams (proxy for rivalry intensity; NaN when insufficient history)
    """
    if not records:
        return pd.DataFrame(
            columns=["match_id", "same_city", "same_region", "historical_intensity"]
        )

    sorted_records = sorted(records, key=lambda r: (r.date, r.match_id))

    # Pre-compute per-pair meeting history
    pair_goals: dict[tuple[str, str], list[int]] = defaultdict(list)

    rows: list[dict] = []
    for rec in sorted_records:
        pair = _pair_key(rec.home_team, rec.away_team)

        # Historical intensity: avg total_goals over last N meetings BEFORE this one
        past = pair_goals[pair]
        if len(past) >= _INTENSITY_WINDOW:
            intensity = sum(past[-_INTENSITY_WINDOW:]) / _INTENSITY_WINDOW
        elif past:
            intensity = sum(past) / len(past)
        else:
            intensity = float("nan")

        rows.append(
            {
                "match_id": rec.match_id,
                "same_city": int(_same_city(rec.home_team, rec.away_team)),
                "same_region": int(_same_region(rec.home_team, rec.away_team)),
                "historical_intensity": intensity,
            }
        )

        # Record this match's total for future lookups
        pair_goals[pair].append(rec.total_goals)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pair_key(team_a: str, team_b: str) -> tuple[str, str]:
    """Canonical sorted pair key so A-vs-B and B-vs-A are the same."""
    return tuple(sorted((team_a, team_b)))


def _same_city(home: str, away: str) -> bool:
    """True if both teams share a known city."""
    city_h = _CITY_MAP.get(home)
    city_a = _CITY_MAP.get(away)
    return city_h is not None and city_h == city_a


def _same_region(home: str, away: str) -> bool:
    """True if both teams share a known country."""
    country_h = _COUNTRY_MAP.get(home)
    country_a = _COUNTRY_MAP.get(away)
    return country_h is not None and country_h == country_a
