"""Constants and enums for the betting model."""

from enum import Enum


class PropType(str, Enum):
    """Player prop types."""
    POINTS = "points"
    REBOUNDS = "rebounds"
    ASSISTS = "assists"
    STEALS = "steals"
    BLOCKS = "blocks"
    THREES = "threes"
    TURNOVERS = "turnovers"
    PRA = "pra"  # points + rebounds + assists
    PR = "pr"    # points + rebounds
    PA = "pa"    # points + assists
    RA = "ra"    # rebounds + assists


# Common betting thresholds for each prop type
PROP_THRESHOLDS = {
    PropType.POINTS: [9.5, 14.5, 19.5, 24.5, 29.5, 34.5, 39.5],
    PropType.REBOUNDS: [3.5, 5.5, 7.5, 9.5, 11.5, 13.5],
    PropType.ASSISTS: [2.5, 4.5, 6.5, 8.5, 10.5, 12.5],
    PropType.STEALS: [0.5, 1.5, 2.5, 3.5],
    PropType.BLOCKS: [0.5, 1.5, 2.5, 3.5],
    PropType.THREES: [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
    PropType.TURNOVERS: [1.5, 2.5, 3.5, 4.5],
    PropType.PRA: [19.5, 24.5, 29.5, 34.5, 39.5, 44.5, 49.5],
    PropType.PR: [14.5, 19.5, 24.5, 29.5, 34.5],
    PropType.PA: [14.5, 19.5, 24.5, 29.5, 34.5],
    PropType.RA: [9.5, 12.5, 15.5, 18.5],
}

# League average stats for baseline comparisons
LEAGUE_AVERAGES = {
    "def_rating": 112.0,  # Points allowed per 100 possessions
    "off_rating": 112.0,  # Points scored per 100 possessions
    "pace": 100.0,        # Possessions per game
}
