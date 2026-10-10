# Spotify Music Recommender with PySpark

A content-based music recommender built with PySpark: songs are grouped by their audio features, and the closest songs inside the same cluster are recommended.

> Notebook written in Portuguese (course material).

## Approach

1. **Exploration:** audio features (danceability, energy, valence, …) by year and genre, plus correlation analysis (Plotly).
2. **Feature pipeline:** `VectorAssembler` → `StandardScaler` → `PCA` with 6 components, chosen to explain about 70% of the variance.
3. **Clustering:** `KMeans` (k = 50) on the PCA projection.
4. **Recommendation:** for a given song, rank the other songs in its cluster by Euclidean distance in PCA space.
5. **Spotify Web API:** fetch album covers and track metadata for the recommendations ([Spotipy](https://spotipy.readthedocs.io/)).

![Recommendation example](image.png)

## Setup

- Apache Spark installed, with `SPARK_HOME` set (the notebook uses `findspark`).
- The course datasets `dados_musicas.csv`, `dados_musicas_ano.csv` and `dados_musicas_genero.csv` placed in `data/`.
- Spotify API credentials from the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard), exported as environment variables:

```bash
export SPOTIPY_CLIENT_ID=...
export SPOTIPY_CLIENT_SECRET=...
pip install -r requirements.txt
jupyter notebook spotify_recommender.ipynb
```
