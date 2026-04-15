"""
Run: python app.py
"""

import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.retrieval import retrieve_songs
from src.recommender import load_songs, recommend_songs

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recommender.log")
SONGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "songs.csv")

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

VALID_GENRES = {"pop", "lofi", "rock", "ambient", "jazz", "synthwave", "indie pop"}
VALID_MOODS = {"happy", "chill", "intense", "relaxed", "moody", "focused"}


def confidence_label(score: float) -> str:
    if score >= 3.5:
        return "High"
    elif score >= 2.0:
        return "Medium"
    else:
        return "Low"


def validate_input(genre: str, mood: str, energy: str) -> list:
    errors = []
    if genre.lower() not in VALID_GENRES:
        errors.append(
            f"Invalid genre '{genre}'. Choose from: {', '.join(sorted(VALID_GENRES))}"
        )
    if mood.lower() not in VALID_MOODS:
        errors.append(
            f"Invalid mood '{mood}'. Choose from: {', '.join(sorted(VALID_MOODS))}"
        )
    try:
        e = float(energy)
        if not (0.0 <= e <= 1.0):
            errors.append("Energy must be between 0.0 and 1.0")
    except ValueError:
        errors.append("Energy must be a number (e.g. 0.7)")
    return errors


def main():
    print("=== Explainable AI Music Recommender ===\n")
    print(f"Valid genres : {', '.join(sorted(VALID_GENRES))}")
    print(f"Valid moods  : {', '.join(sorted(VALID_MOODS))}")
    print()

    genre = input("Enter genre  : ").strip()
    mood = input("Enter mood   : ").strip()
    energy = input("Enter energy (0.0–1.0): ").strip()

    errors = validate_input(genre, mood, energy)
    if errors:
        for err in errors:
            print(f"[ERROR] {err}")
        logging.warning(f"Invalid input — genre={genre}, mood={mood}, energy={energy} — {errors}")
        sys.exit(1)

    user_prefs = {"genre": genre.lower(), "mood": mood.lower(), "energy": float(energy)}
    logging.info(f"Input: {user_prefs}")

    # --- RAG Pipeline ---
    all_songs = load_songs(SONGS_PATH)

    # Step 1: Retrieve relevant candidates
    candidates = retrieve_songs(user_prefs, all_songs, top_n=6)
    print(f"\nRetrieved {len(candidates)} candidate(s) matching genre/mood...")

    # Step 2: Score + explain using retrieved data only
    recommendations = recommend_songs(user_prefs, candidates, k=3)

    if not recommendations:
        print("\nNo strong matches found. Try a different genre or mood.")
        logging.info("No recommendations returned.")
        return

    # Step 3: Output
    print("\nTop 3 Recommendations:\n")
    for i, (song, score, explanation) in enumerate(recommendations, 1):
        confidence = confidence_label(score)
        print(f"{i}. {song['title']} by {song['artist']}")
        print(f"   Score: {score:.2f} [{confidence} confidence]")
        print(f"   Because: {explanation}\n")

    logging.info(
        f"Output: {[(r[0]['title'], round(r[1], 2)) for r in recommendations]}"
    )


if __name__ == "__main__":
    main()
