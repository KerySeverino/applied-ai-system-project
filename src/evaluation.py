"""
Automated evaluation for the Explainable AI Music Recommender.

Runs 6 test cases covering normal use, edge cases, and fallback behavior.
Logs all inputs, outputs, and results to evaluation.log.
"""

import logging
import os
import sys

# Allow running as: python src/evaluation.py from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommender import load_songs, recommend_songs

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation.log")
SONGS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "songs.csv")

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def confidence_label(score: float) -> str:
    if score >= 3.5:
        return "High"
    elif score >= 2.0:
        return "Medium"
    else:
        return "Low"


TEST_CASES = [
    {
        "name": "Pop + happy + high energy",
        "user_prefs": {"genre": "pop", "mood": "happy", "energy": 0.8},
        "expect_genre": "pop",
        "expect_mood": "happy",
    },
    {
        "name": "Lofi + chill + low energy",
        "user_prefs": {"genre": "lofi", "mood": "chill", "energy": 0.4},
        "expect_genre": "lofi",
        "expect_mood": "chill",
    },
    {
        "name": "Rock + intense + very high energy",
        "user_prefs": {"genre": "rock", "mood": "intense", "energy": 0.9},
        "expect_genre": "rock",
        "expect_mood": "intense",
    },
    {
        "name": "Jazz + relaxed + low energy",
        "user_prefs": {"genre": "jazz", "mood": "relaxed", "energy": 0.4},
        "expect_genre": "jazz",
        "expect_mood": "relaxed",
    },
    {
        "name": "Ambient + chill + very low energy",
        "user_prefs": {"genre": "ambient", "mood": "chill", "energy": 0.3},
        "expect_genre": "ambient",
        "expect_mood": "chill",
    },
    {
        "name": "Fallback: synthwave + relaxed (no exact match in dataset)",
        "user_prefs": {"genre": "synthwave", "mood": "relaxed", "energy": 0.6},
        "expect_genre": None,  # no exact match — just verify results are returned
        "expect_mood": None,
    },
]


def run_tests() -> tuple:
    songs = load_songs(SONGS_PATH)
    passed = 0
    total = len(TEST_CASES)
    results = []

    for tc in TEST_CASES:
        recs = recommend_songs(tc["user_prefs"], songs, k=3)
        logging.info(f"Input: {tc['user_prefs']}")

        if not recs:
            ok = False
            note = "No results returned"
        elif tc["expect_genre"] is None:
            # Fallback test: just verify something is returned
            ok = len(recs) > 0
            top_song, top_score, _ = recs[0]
            note = f"Fallback returned '{top_song['title']}' score={top_score:.2f} [{confidence_label(top_score)}]"
        else:
            top_song, top_score, _ = recs[0]
            ok = (
                str(top_song["genre"]).lower() == tc["expect_genre"]
                and str(top_song["mood"]).lower() == tc["expect_mood"]
            )
            note = (
                f"Top: '{top_song['title']}' "
                f"({top_song['genre']}/{top_song['mood']}) "
                f"score={top_score:.2f} [{confidence_label(top_score)}]"
            )

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1

        results.append((tc["name"], status, note))
        logging.info(f"Test '{tc['name']}': {status} — {note}")

    # Print summary
    print("\n=== Evaluation Results ===\n")
    for name, status, note in results:
        marker = "✓" if status == "PASS" else "✗"
        print(f"  [{marker}] {name}")
        print(f"       {note}\n")

    summary = f"{passed}/{total} tests passed"
    if passed < total:
        summary += f", struggled with {total - passed} case(s) — limited dataset coverage"
    else:
        summary += ", all cases handled correctly"

    print(f"Summary: {summary}\n")
    logging.info(f"Summary: {summary}")

    return passed, total


if __name__ == "__main__":
    run_tests()
