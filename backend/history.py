from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DATA_DIR = Path(__file__).resolve().parent
HISTORY_PATH = DATA_DIR / "user_history.json"
LIKES_PATH = DATA_DIR / "user_likes.json"
MAX_HISTORY = 50
MAX_LIKES = 100


def _ensure_history_file() -> None:
    if not HISTORY_PATH.exists():
        HISTORY_PATH.write_text("[]", encoding="utf-8")
    if not LIKES_PATH.exists():
        LIKES_PATH.write_text("[]", encoding="utf-8")


def load_history() -> List[Dict]:
    _ensure_history_file()
    with HISTORY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_likes() -> List[Dict]:
    _ensure_history_file()
    with LIKES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_history(history: List[Dict]) -> None:
    with HISTORY_PATH.open("w", encoding="utf-8") as f:
        json.dump(history[-MAX_HISTORY:], f, ensure_ascii=False, indent=2)


def save_likes(likes: List[Dict]) -> None:
    with LIKES_PATH.open("w", encoding="utf-8") as f:
        json.dump(likes[-MAX_LIKES:], f, ensure_ascii=False, indent=2)


def add_history_entry(emotion: str, songs: List[Dict], image_base64: str) -> None:
    history = load_history()
    history.append(
        {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "emotion": emotion,
            "songs": songs,
            "image": image_base64,
        }
    )
    save_history(history)


def get_history() -> List[Dict]:
    history = load_history()
    history.reverse()
    return history


def add_liked_song(song: Dict) -> None:
    likes = load_likes()
    song_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "Song": song.get("Song"),
        "Album": song.get("Album"),
        "Artist": song.get("Artist"),
        "Type": song.get("Type"),
    }
    likes.append(song_entry)
    save_likes(likes)


def get_liked_songs() -> List[Dict]:
    likes = load_likes()
    likes.reverse()
    return likes


def get_most_frequent_emotion(history: Optional[List[Dict]] = None) -> Optional[str]:
    if history is None:
        history = load_history()
    emotions = [entry.get("emotion") for entry in history if entry.get("emotion")]
    if not emotions:
        return None
    counter = Counter(emotions)
    most_common = counter.most_common(1)[0][0]
    return most_common


def get_profile_data() -> Tuple[List[Dict], Optional[str], List[Dict]]:
    history = get_history()
    most_frequent = get_most_frequent_emotion(history)
    return history, most_frequent, get_liked_songs()

