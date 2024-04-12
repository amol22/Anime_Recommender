import sys
import os
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.components.config import DataTransformationConfig
from scipy.sparse import csr_matrix

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def initiate_data_transformation(self,rating_df_path):
        logging.info("Data transformation initiated")
        try:
            rating_df = pd.read_csv(rating_df_path)
            
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
            
            csr_data = csr_matrix(rating_df_pivot.values)
            rating_df_pivot.reset_index(inplace=True)
            
            os.makedirs(os.path.dirname(self.data_transformation_config.rating_df_pivot_path), exist_ok  = True)
            rating_df_pivot.to_csv(self.data_transformation_config.rating_df_pivot_path, index = False, header = True)
            
            logging.info("Data pivot filtered")
            
            return csr_data
            
        except Exception as e:
            raise CustomException(e,sys)
        
        