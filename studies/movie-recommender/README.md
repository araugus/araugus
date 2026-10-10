# Movie Recommender (MovieLens)

An introduction to recommender systems using simple, explainable heuristics on the MovieLens dataset (86k movies, 33M ratings).

> Notebook written in Portuguese (course material).

## Approach

1. **Most-voted movies:** rank movies by number of ratings.
2. **Highest average rating:** rank by mean rating, filtered by a minimum number of votes so movies with only a few ratings don't dominate the top.
3. **Similar to what I watched:** filter by the genres of movies already watched and exclude those movies from the results.

## Data

[MovieLens Latest (full)](https://grouplens.org/datasets/movielens/latest/). Download it and place `movies.csv` and `ratings.csv` in `data/`.

## How to run

```bash
pip install pandas jupyter
jupyter notebook movie_recommendation.ipynb
```
