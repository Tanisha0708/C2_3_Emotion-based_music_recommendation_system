from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parent
SONGS_PATH = DATA_DIR / "songs.csv"


class KNNMusicRecommender:
    """
    Train a similarity-based recommender on curated song metadata and expose
    helper utilities for fetching tailored playlists from emotion cues.
    """

    feature_columns = ["Tempo", "Energy", "Danceability", "Genre_encoded"]

    def __init__(self):
        if not SONGS_PATH.exists():
            raise FileNotFoundError(f"Songs dataset not found at {SONGS_PATH}")

        self.df = pd.read_csv(SONGS_PATH)
        self.label_encoders: Dict[str, LabelEncoder] = {}

        # Encode categorical columns
        for column in ["Genre", "MoodLabel"]:
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(self.df[column])
            self.df[f"{column}_encoded"] = encoded
            self.label_encoders[column] = encoder

        # Prepare scaler and similarity model
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.df[self.feature_columns])

        self.knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
        self.knn.fit(self.scaled_features)
        self.recommendation_count = min(10, len(self.df))

        self.emotion_to_features = self._build_emotion_feature_map()

    def _build_emotion_feature_map(self) -> Dict[str, Sequence[float]]:
        """
        Map each supported emotion to a representative numeric feature vector.
        Tempo is in BPM, energy/danceability are normalized (0-1), and genre is
        expressed via the trained LabelEncoder to ensure consistency.
        """
        genre_encoder = self.label_encoders["Genre"]

        blueprint = {
            "Happy": (132, 0.84, 0.9, "Pop"),
            "Sad": (70, 0.36, 0.45, "Acoustic"),
            "Angry": (146, 0.93, 0.57, "Rock"),
            "Disgusted": (138, 0.8, 0.5, "Metal"),
            "Fearful": (80, 0.44, 0.38, "Ambient"),
            "Neutral": (108, 0.54, 0.6, "Indie"),
            "Surprised": (136, 0.82, 0.83, "Electronic"),
        }

        numeric_map: Dict[str, Sequence[float]] = {}
        for emotion, (tempo, energy, danceability, genre_label) in blueprint.items():
            if genre_label not in genre_encoder.classes_:
                raise ValueError(
                    f"Genre '{genre_label}' for emotion '{emotion}' not present in dataset."
                )
            genre_value = float(genre_encoder.transform([genre_label])[0])
            numeric_map[emotion] = [tempo, energy, danceability, genre_value]

        return numeric_map

    def _get_feature_vector(self, emotion: str) -> np.ndarray:
        emotion_key = emotion if emotion in self.emotion_to_features else "Neutral"
        vector = np.array(self.emotion_to_features[emotion_key], dtype=float)
        return vector.reshape(1, -1)

    def recommend(self, emotion: str) -> List[Dict[str, str]]:
        vector = self._get_feature_vector(emotion)
        scaled_vector = self.scaler.transform(vector)
        distances, indices = self.knn.kneighbors(
            scaled_vector, n_neighbors=self.recommendation_count
        )

        result_rows = self.df.iloc[indices[0]][
            ["Song", "Artist", "Album", "Genre", "SongType", "MoodLabel"]
        ]
        recommendations = []
        for _, row in result_rows.iterrows():
            recommendations.append(
                {
                    "Song": row["Song"],
                    "Artist": row["Artist"],
                    "Album": row.get("Album", "-"),
                    "Genre": row["Genre"],
                    "Type": row.get("SongType", row["Genre"]),
                    "MoodLabel": row["MoodLabel"],
                }
            )
        return recommendations

    def recommend_dataframe(self, emotion: str) -> pd.DataFrame:
        records = self.recommend(emotion)
        if not records:
            return pd.DataFrame(columns=["Name", "Album", "Artist", "Type"])
        df = pd.DataFrame(records)
        df["Name"] = df["Song"]
        df["Album"] = df.get("Album", df.get("MoodLabel", "-"))
        df["Artist"] = df["Artist"]
        df["Type"] = df.get("Type", df.get("Genre", ""))
        return df[["Name", "Album", "Artist", "Type"]]


_MUSIC_RECOMMENDER: Optional[KNNMusicRecommender] = None


def get_recommender() -> KNNMusicRecommender:
    global _MUSIC_RECOMMENDER
    if _MUSIC_RECOMMENDER is None:
        _MUSIC_RECOMMENDER = KNNMusicRecommender()
    return _MUSIC_RECOMMENDER


def recommend_songs_ml(emotion: str):
    return get_recommender().recommend(emotion)


def recommend_songs_dataframe(emotion: str):
    return get_recommender().recommend_dataframe(emotion)

