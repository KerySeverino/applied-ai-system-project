"""
Retrieval module for the AI Music Recommender.

This is the RAG retrieval step: given user preferences,
fetch the most relevant candidate songs from the dataset
before scoring and explanation happen.
"""

from typing import List, Dict


def retrieve_songs(user_prefs: Dict, songs: List[Dict], top_n: int = 6) -> List[Dict]:
    """
    Retrieve candidate songs based on genre and mood match.

    Strategy:
      1. Exact match on both genre and mood
      2. Partial match on either genre or mood
      3. Fill remaining slots with the rest of the catalog

    Returns up to top_n candidates for downstream scoring.
    """
    genre = user_prefs.get("genre", "").lower()
    mood = user_prefs.get("mood", "").lower()

    exact = [
        s for s in songs
        if str(s["genre"]).lower() == genre and str(s["mood"]).lower() == mood
    ]
    partial = [
        s for s in songs
        if s not in exact and (
            str(s["genre"]).lower() == genre or str(s["mood"]).lower() == mood
        )
    ]
    rest = [s for s in songs if s not in exact and s not in partial]

    candidates = (exact + partial + rest)[:top_n]
    return candidates
