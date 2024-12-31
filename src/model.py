import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)

class MovieRecommender:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)  # Reset index to ensure sequential numbering
        self.similarity_matrix = None

    def build_similarity_matrix(self):
        """
        Build a similarity matrix using a simple content-based approach
        based on 'genre'.
        """
        if 'genre' not in self.df.columns:
            logging.warning("DataFrame does not have 'genre' column. Similarity not computed.")
            self.similarity_matrix = None
            return

        # Convert genre strings to dummy variables: "Action|Comedy" => multiple columns
        genre_dummies = self.df['genre'].str.get_dummies('|')

        # Compute cosine similarity on these genre vectors
        self.similarity_matrix = cosine_similarity(genre_dummies.values)
        logging.info("Similarity matrix built using genre information.")

    def get_recommendations(self, movie_title: str, top_n: int = 5) -> pd.DataFrame:
        """
        Return top N recommended movies based on similarity to the given movie title.
        """
        # Check if we have a valid similarity matrix
        if self.similarity_matrix is None:
            logging.error("Similarity matrix has not been built. Returning empty DataFrame.")
            return pd.DataFrame()

        # Find index of the movie title
        indices = self.df[self.df['title'] == movie_title].index
        if len(indices) == 0:
            logging.warning(f"Movie title '{movie_title}' not found in the dataset.")
            return pd.DataFrame()

        movie_idx = indices[0]

        # Calculate similarity scores for the selected movie
        sim_scores = list(enumerate(self.similarity_matrix[movie_idx]))

        # Sort by similarity in descending order
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # The first item is the queried movie itself, so skip it
        sim_scores = sim_scores[1: top_n + 1]

        # Get the movie indices for top recommendations
        recommended_indices = [i[0] for i in sim_scores]
        recommended_movies = self.df.iloc[recommended_indices]

        # Return relevant columns
        return recommended_movies[['title', 'genre', 'year', 'rating']].reset_index(drop=True)
