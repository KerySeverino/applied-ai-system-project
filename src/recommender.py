from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def _score(self, user: UserProfile, song: Song) -> float:
        s = 0.0
        if song.genre == user.favorite_genre:
            s += 2.0
        if song.mood == user.favorite_mood:
            s += 1.5
        s += 1.0 * (1.0 - abs(user.target_energy - song.energy))
        if user.likes_acoustic and song.acousticness > 0.5:
            s += 0.5
        return s

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        return sorted(self.songs, key=lambda s: self._score(user, s), reverse=True)[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        reasons = []
        if song.genre == user.favorite_genre:
            reasons.append(f"matches your preferred genre ({song.genre})")
        if song.mood == user.favorite_mood:
            reasons.append(f"fits your preferred mood ({song.mood})")
        if abs(user.target_energy - song.energy) <= 0.15:
            reasons.append(f"has a close energy level ({song.energy:.2f})")
        if user.likes_acoustic and song.acousticness > 0.5:
            reasons.append(f"is acoustic-leaning ({song.acousticness:.2f} acousticness)")
        if not reasons:
            reasons.append(f"is a {song.genre} {song.mood} track with energy {song.energy:.2f}")
        return "This song " + ", and ".join(reasons) + "."


def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    import pandas as pd
    print(f"Loading songs from {csv_path}...")
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional RAG implementation: retrieve candidates by genre/mood,
    then score and rank them.
    Returns list of (song_dict, score, explanation) tuples.
    """
    genre = user_prefs.get("genre", "").lower()
    mood = user_prefs.get("mood", "").lower()
    target_energy = float(user_prefs.get("energy", 0.5))

    # --- Retrieval step ---
    exact = [s for s in songs if str(s["genre"]).lower() == genre and str(s["mood"]).lower() == mood]
    partial = [s for s in songs if s not in exact and (str(s["genre"]).lower() == genre or str(s["mood"]).lower() == mood)]
    rest = [s for s in songs if s not in exact and s not in partial]
    candidates = (exact + partial + rest)[:max(k * 2, 6)]

    # --- Scoring step ---
    scored = []
    for song in candidates:
        s = 0.0
        if str(song["genre"]).lower() == genre:
            s += 2.0
        if str(song["mood"]).lower() == mood:
            s += 1.5
        s += 1.0 * (1.0 - abs(target_energy - float(song["energy"])))

        # --- Explanation step (grounded in retrieved song data) ---
        parts = []
        if str(song["genre"]).lower() == genre:
            parts.append(f"matches your genre ({song['genre']})")
        if str(song["mood"]).lower() == mood:
            parts.append(f"fits your mood ({song['mood']})")
        energy_diff = abs(target_energy - float(song["energy"]))
        if energy_diff <= 0.2:
            parts.append(f"has similar energy ({float(song['energy']):.2f})")
        if "description" in song and song["description"]:
            parts.append(str(song["description"]))
        if not parts:
            parts.append(f"is a {song['genre']} {song['mood']} track with energy {float(song['energy']):.2f}")

        scored.append((song, s, "; ".join(parts)))

    scored.sort(key=lambda x: x[1], reverse=True)

    if not scored or scored[0][1] < 1.0:
        return []

    return scored[:k]
