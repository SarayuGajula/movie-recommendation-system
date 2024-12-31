import streamlit as st
import pandas as pd
from src.data_preprocessing import load_data, clean_data
from src.model import MovieRecommender
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)

def main():
    st.title("Movie Recommendation System")
    st.write("""
    This application demonstrates a simple content-based recommendation system using movie genres.
    """)

    # Load and clean data
    try:
        df = load_data("data/movies.csv")
    except FileNotFoundError:
        st.error("Could not find 'data/movies.csv'. Please ensure the file exists.")
        return

    df = clean_data(df)

    # Build recommendation system
    recommender = MovieRecommender(df)
    recommender.build_similarity_matrix()

    # Movie selector
    if len(df) > 0:
        movie_list = df['title'].unique().tolist()
        selected_movie = st.selectbox("Choose a movie to get recommendations:", movie_list)

        top_n = st.slider("Number of recommendations:", min_value=1, max_value=5, value=3)

        # Button to trigger recommendation
        if st.button("Recommend"):
            recommendations = recommender.get_recommendations(selected_mov
