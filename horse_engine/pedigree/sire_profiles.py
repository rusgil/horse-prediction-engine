"""
Curated Australian/NZ sire profiles.

Each entry encodes the typical distance aptitude, wet-track affinity,
first-up readiness, and running style tendency of the sire line.
This is used when live pedigree data is unavailable and as a prior.

Keys in each record:
  dist_min / dist_max   — ideal distance window (metres)
  aptitude              — "sprint" | "mile" | "middle" | "staying"
  wet                   — 0–10 wet-track affinity (10 = loves it)
  first_up              — 0–10 first-up readiness
  on_pace               — 0–10 tendency to lead or be on speed
  brilliance            — 0–10 explosive turn-of-foot (sprint)
  stamina               — 0–10 staying power
  dosage_index          — classic pedigree speed:stamina ratio (>4 sprint, <1 staying)
"""

SIRE_PROFILES: dict[str, dict] = {
    # ── Elite Australian Sprint Sires ────────────────────────────────────
    "I Am Invincible": {
        "dist_min": 1000, "dist_max": 1400, "aptitude": "sprint",
        "wet": 4, "first_up": 9, "on_pace": 8, "brilliance": 10, "stamina": 2,
        "dosage_index": 6.8, "note": "Dominant sprint sire; offspring fire first-up and love to lead"
    },
    "Snitzel": {
        "dist_min": 1000, "dist_max": 1400, "aptitude": "sprint",
        "wet": 6, "first_up": 8, "on_pace": 7, "brilliance": 9, "stamina": 3,
        "dosage_index": 5.9, "note": "Versatile speed; reasonable wet-tracker"
    },
    "Exceed And Excel": {
        "dist_min": 1000, "dist_max": 1200, "aptitude": "sprint",
        "wet": 3, "first_up": 7, "on_pace": 9, "brilliance": 10, "stamina": 1,
        "dosage_index": 8.2, "note": "Pure sprint speed; preference for firm ground"
    },
    "Fastnet Rock": {
        "dist_min": 1000, "dist_max": 1600, "aptitude": "sprint",
        "wet": 5, "first_up": 8, "on_pace": 7, "brilliance": 9, "stamina": 4,
        "dosage_index": 5.2, "note": "Brilliant sprint/mile sire; versatile in conditions"
    },
    "Rubick": {
        "dist_min": 1000, "dist_max": 1200, "aptitude": "sprint",
        "wet": 4, "first_up": 8, "on_pace": 9, "brilliance": 9, "stamina": 2,
        "dosage_index": 7.1, "note": "On-pace speed machines"
    },
    "Zoustar": {
        "dist_min": 1000, "dist_max": 1400, "aptitude": "sprint",
        "wet": 5, "first_up": 7, "on_pace": 7, "brilliance": 8, "stamina": 3,
        "dosage_index": 5.5, "note": "Versatile speed; good wet-ground record"
    },
    "Spirit of Boom": {
        "dist_min": 1000, "dist_max": 1200, "aptitude": "sprint",
        "wet": 5, "first_up": 8, "on_pace": 8, "brilliance": 8, "stamina": 2,
        "dosage_index": 6.4, "note": "Queensland speed specialist"
    },
    "Written Tycoon": {
        "dist_min": 1000, "dist_max": 1400, "aptitude": "sprint",
        "wet": 6, "first_up": 7, "on_pace": 6, "brilliance": 8, "stamina": 4,
        "dosage_index": 4.9, "note": "Good wet performer; handles range of distances"
    },
    "Wandjina": {
        "dist_min": 1000, "dist_max": 1200, "aptitude": "sprint",
        "wet": 5, "first_up": 7, "on_pace": 8, "brilliance": 9, "stamina": 2,
        "dosage_index": 7.3,
    },
    "All Too Hard": {
        "dist_min": 1200, "dist_max": 1600, "aptitude": "sprint",
        "wet": 5, "first_up": 6, "on_pace": 6, "brilliance": 8, "stamina": 5,
        "dosage_index": 4.8,
    },
    "Not A Single Doubt": {
        "dist_min": 1000, "dist_max": 1400, "aptitude": "sprint",
        "wet": 4, "first_up": 8, "on_pace": 7, "brilliance": 9, "stamina": 3,
        "dosage_index": 5.6,
    },
    "Shooting To Win": {
        "dist_min": 1000, "dist_max": 1400, "aptitude": "sprint",
        "wet": 5, "first_up": 7, "on_pace": 8, "brilliance": 8, "stamina": 3,
        "dosage_index": 5.8,
    },
    "Foxwedge": {
        "dist_min": 1000, "dist_max": 1400, "aptitude": "sprint",
        "wet": 4, "first_up": 7, "on_pace": 7, "brilliance": 8, "stamina": 3,
        "dosage_index": 5.9,
    },
    "Choisir": {
        "dist_min": 1000, "dist_max": 1300, "aptitude": "sprint",
        "wet": 4, "first_up": 7, "on_pace": 8, "brilliance": 9, "stamina": 2,
        "dosage_index": 7.0,
    },

    # ── Mile / Middle Distance ────────────────────────────────────────────
    "Dundeel": {
        "dist_min": 1400, "dist_max": 2000, "aptitude": "middle",
        "wet": 8, "first_up": 5, "on_pace": 5, "brilliance": 6, "stamina": 7,
        "dosage_index": 2.4, "note": "Outstanding wet-track sire; horses improve with racing"
    },
    "Epaulette": {
        "dist_min": 1200, "dist_max": 1800, "aptitude": "mile",
        "wet": 5, "first_up": 6, "on_pace": 6, "brilliance": 7, "stamina": 5,
        "dosage_index": 3.8,
    },
    "Pierro": {
        "dist_min": 1200, "dist_max": 1600, "aptitude": "mile",
        "wet": 5, "first_up": 7, "on_pace": 5, "brilliance": 8, "stamina": 5,
        "dosage_index": 4.3, "note": "High-quality mile sire with strong first-up record"
    },
    "Lonhro": {
        "dist_min": 1400, "dist_max": 2000, "aptitude": "middle",
        "wet": 6, "first_up": 6, "on_pace": 6, "brilliance": 7, "stamina": 6,
        "dosage_index": 3.2,
    },
    "Street Boss": {
        "dist_min": 1200, "dist_max": 1600, "aptitude": "mile",
        "wet": 5, "first_up": 6, "on_pace": 7, "brilliance": 7, "stamina": 5,
        "dosage_index": 4.1,
    },
    "Smart Missile": {
        "dist_min": 1200, "dist_max": 1600, "aptitude": "mile",
        "wet": 5, "first_up": 6, "on_pace": 6, "brilliance": 8, "stamina": 4,
        "dosage_index": 4.7,
    },
    "Redoute's Choice": {
        "dist_min": 1200, "dist_max": 1800, "aptitude": "mile",
        "wet": 5, "first_up": 6, "on_pace": 6, "brilliance": 8, "stamina": 5,
        "dosage_index": 4.2, "note": "Champion sire across all conditions"
    },
    "More Than Ready": {
        "dist_min": 1200, "dist_max": 1800, "aptitude": "mile",
        "wet": 5, "first_up": 6, "on_pace": 6, "brilliance": 8, "stamina": 5,
        "dosage_index": 4.0,
    },
    "Mossman": {
        "dist_min": 1000, "dist_max": 1600, "aptitude": "mile",
        "wet": 6, "first_up": 7, "on_pace": 7, "brilliance": 7, "stamina": 4,
        "dosage_index": 4.5,
    },
    "Manhattan Rain": {
        "dist_min": 1400, "dist_max": 2000, "aptitude": "middle",
        "wet": 7, "first_up": 5, "on_pace": 5, "brilliance": 6, "stamina": 7,
        "dosage_index": 2.6,
    },
    "High Chaparral": {
        "dist_min": 1600, "dist_max": 2400, "aptitude": "staying",
        "wet": 9, "first_up": 4, "on_pace": 4, "brilliance": 5, "stamina": 9,
        "dosage_index": 1.2, "note": "Premier wet-track staying sire"
    },
    "Darci Brahma": {
        "dist_min": 1400, "dist_max": 2000, "aptitude": "middle",
        "wet": 7, "first_up": 5, "on_pace": 5, "brilliance": 6, "stamina": 7,
        "dosage_index": 2.8, "note": "Strong NZ-bred wet-trackers"
    },

    # ── Stayers ──────────────────────────────────────────────────────────
    "Galileo": {
        "dist_min": 2000, "dist_max": 3200, "aptitude": "staying",
        "wet": 7, "first_up": 3, "on_pace": 4, "brilliance": 5, "stamina": 10,
        "dosage_index": 0.9, "note": "World's leading staying sire; needs runs"
    },
    "Frankel": {
        "dist_min": 1600, "dist_max": 2400, "aptitude": "staying",
        "wet": 6, "first_up": 4, "on_pace": 5, "brilliance": 7, "stamina": 8,
        "dosage_index": 1.8,
    },
    "Savabeel": {
        "dist_min": 1600, "dist_max": 2400, "aptitude": "staying",
        "wet": 8, "first_up": 4, "on_pace": 4, "brilliance": 5, "stamina": 9,
        "dosage_index": 1.4, "note": "Dominant NZ staying sire; thrives on rain-affected going"
    },
    "Ocean Park": {
        "dist_min": 1600, "dist_max": 2400, "aptitude": "staying",
        "wet": 8, "first_up": 4, "on_pace": 4, "brilliance": 5, "stamina": 9,
        "dosage_index": 1.6, "note": "NZ staying sire; wet-track specialist"
    },
    "So You Think": {
        "dist_min": 1600, "dist_max": 2500, "aptitude": "staying",
        "wet": 8, "first_up": 4, "on_pace": 5, "brilliance": 6, "stamina": 9,
        "dosage_index": 1.5, "note": "Multiple Cox Plate winner; exceptional wet-track performer"
    },
    "Fiorente": {
        "dist_min": 1800, "dist_max": 3200, "aptitude": "staying",
        "wet": 8, "first_up": 3, "on_pace": 3, "brilliance": 4, "stamina": 10,
        "dosage_index": 0.8,
    },
    "Reliable Man": {
        "dist_min": 1600, "dist_max": 2800, "aptitude": "staying",
        "wet": 7, "first_up": 4, "on_pace": 4, "brilliance": 5, "stamina": 9,
        "dosage_index": 1.3,
    },
    "Shamardal": {
        "dist_min": 1600, "dist_max": 2000, "aptitude": "middle",
        "wet": 5, "first_up": 5, "on_pace": 6, "brilliance": 7, "stamina": 6,
        "dosage_index": 3.1,
    },
    "Medaglia d'Oro": {
        "dist_min": 1400, "dist_max": 2000, "aptitude": "middle",
        "wet": 5, "first_up": 5, "on_pace": 5, "brilliance": 7, "stamina": 6,
        "dosage_index": 2.9,
    },
    "Fastnet Rock": {  # duplicate key below handled by dict — keep last entry
        "dist_min": 1000, "dist_max": 1600, "aptitude": "sprint",
        "wet": 5, "first_up": 8, "on_pace": 7, "brilliance": 9, "stamina": 4,
        "dosage_index": 5.2,
    },
}

# Normalise keys to lowercase for fuzzy matching
_SIRE_LOWER: dict[str, dict] = {k.lower(): v for k, v in SIRE_PROFILES.items()}


def get_sire_profile(sire_name: str) -> dict | None:
    """Case-insensitive lookup; returns None if sire not in our database."""
    return _SIRE_LOWER.get(sire_name.lower())


def get_distance_aptitude(sire_name: str, race_distance: int) -> float:
    """
    0–1 score: how well this sire's offspring handle race_distance.
    1.0 = perfect match, 0 = way outside aptitude.
    """
    profile = get_sire_profile(sire_name)
    if not profile:
        return 0.5  # unknown sire: neutral prior

    lo, hi = profile["dist_min"], profile["dist_max"]
    ideal_mid = (lo + hi) / 2
    window = (hi - lo) / 2 + 200  # 200m grace either side

    if lo <= race_distance <= hi:
        return 1.0

    dist_from_ideal = abs(race_distance - ideal_mid)
    score = max(0.0, 1.0 - (dist_from_ideal - window) / 600)
    return round(score, 3)


def get_wet_track_pedigree_score(sire_name: str, dam_sire_name: str, condition_category: str) -> float:
    """
    Combined wet-track score from sire AND dam's sire (broodmare influence).
    Returns 0–10.
    """
    sire_p = get_sire_profile(sire_name)
    dam_p = get_sire_profile(dam_sire_name)

    sire_wet = sire_p["wet"] if sire_p else 5.0
    dam_wet = dam_p["wet"] if dam_p else 5.0

    # Sire contributes 60%, dam's sire 40%
    combined = sire_wet * 0.6 + dam_wet * 0.4

    # Only relevant on soft/heavy
    if condition_category in ("soft", "heavy"):
        return round(combined, 2)
    return 5.0  # neutral on good ground


CONDITION_CATEGORY = {
    "Firm 1": "firm", "Firm 2": "firm",
    "Good 3": "good", "Good 4": "good",
    "Soft 5": "soft", "Soft 6": "soft", "Soft 7": "soft",
    "Heavy 8": "heavy", "Heavy 9": "heavy", "Heavy 10": "heavy",
    "Synthetic": "synthetic",
}


def parse_condition_category(track_condition: str) -> str:
    return CONDITION_CATEGORY.get(track_condition, "good")
