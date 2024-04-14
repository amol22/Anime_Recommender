import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import load_npz

class SuggestionPipeline:
    def __init__(self):
        pass
    
    def get_suggestions(self,anime,num,anime_df,rating_df_pivot,csr_data):
        try:
            model_path = "artifacts/model.pkl"
            knn = load_object(model_path)
            name=anime_df[anime_df['name']==anime]
            loc=name.iloc[0]['anime_id']
            index=rating_df_pivot[rating_df_pivot['anime_id'] == loc].index[0]
            distances , indices = knn.kneighbors(csr_data[index],n_neighbors=num+1)
            rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),\
                                key=lambda x: x[1],reverse=True)[0:10]
            recommend_frame = []
            for val in rec_movie_indices:
                    anime_idx = rating_df_pivot.iloc[val[0]]['anime_id']
                    idx = anime_df[anime_df['anime_id'] == anime_idx].index
                    recommend_frame.append({'Title':anime_df.iloc[idx]['name'].values[0],'Type':anime_df.iloc[idx]['type'].values[0],'Rating':anime_df.iloc[idx]['rating'].values[0]})
            df = pd.DataFrame(recommend_frame,index=range(1,11))
            return df
        except Exception as e:
            raise CustomException(e,sys)

if __name__=='__main__':
    logging.info('begin')
    rating_df = pd.read_csv('artifacts/rating_data.csv')
    anime_df = pd.read_csv('artifacts/anime_data.csv')
    csr_data = load_npz('artifacts/csr_data.csv.npz')
            
    rating_df.drop_duplicates(subset=['anime_id','user_id'],inplace=True)
    rating_df_pivot = rating_df.pivot(index='anime_id',columns='user_id',values='rating')
    rating_df_pivot.fillna(0,inplace=True)
    rating_df_pivot.replace(-1.0,0,inplace=True)
    logging.info("Data pivot created")
    anime_votes = rating_df.groupby('anime_id')['rating'].agg('count')
    anime_filter = anime_votes[anime_votes>5]
    users = rating_df.groupby('user_id')['rating'].agg('count')
    user_filter = users[users>18]
                    
    rating_df_pivot = rating_df_pivot.loc[anime_filter.index,:]
    rating_df_pivot = rating_df_pivot.loc[:,user_filter.index]
    rating_df_pivot.reset_index(inplace=True)
    SP=SuggestionPipeline()
    res=SP.get_suggestions('Haikyuu!! Second Season',10)
    logging.info('end')


        