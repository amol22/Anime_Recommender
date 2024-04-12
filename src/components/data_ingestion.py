import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from src.components.config import DataIngestionConfig
from src.components.data_transformation import DataTransformation

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig
        
    def data_ingestion(self):
        logging.info("Data ingestion initiated")
        try:
            anime_df = pd.read_csv('data/anime.csv')
            rating_df = pd.read_csv('data/rating.csv')
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_anime_data_path), exist_ok  = True)
            anime_df.to_csv(self.ingestion_config.raw_anime_data_path, index = False, header = True)
            os.makedirs(os.path.dirname(self.ingestion_config.raw_rating_data_path), exist_ok= True)
            rating_df.to_csv(self.ingestion_config.raw_rating_data_path, index = False, header = True)
            
            logging.info("Data ingestion completed")
            
            return (
                self.ingestion_config.raw_anime_data_path,
                self.ingestion_config.raw_rating_data_path
            )
        except Exception as e:
            raise CustomException(e,sys) 
        
if __name__ == '__main__':
    DI = DataIngestion()
    x,y = DI.data_ingestion()
    DT = DataTransformation()
    tdf = DT.initiate_data_transformation(y)
