import streamlit as st
from src.pipelines.suggestion_pipeline import SuggestionPipeline
import pandas as pd
from scipy.sparse import load_npz

ls = pd.read_csv('artifacts/filtered_anime_list.csv')
a_list = ls['name'].to_list()

rating_df_pivot = pd.read_parquet('artifacts/rating_df_pivot.csv')
anime_df = pd.read_csv('artifacts/anime_data.csv')
csr_data = load_npz('artifacts/csr_data.csv.npz')

def main():
    st.title("Anime Recommender System")
    st.write("Do you struggle with finding animes similar to ones that you have watched? Worry not, just select the anime/movie below and we will recommend you similar anime.")
    anime = st.selectbox('Select Anime',a_list)
    if st.button('Suggest me!'):
        SP=SuggestionPipeline()
        result=SP.get_suggestions(anime,10,anime_df,rating_df_pivot,csr_data)
        st.dataframe(result)
        
if __name__ == "__main__":
    main()