import os
import sys
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_rating_data_path: str = os.path.join('artifacts','rating_data.csv')
    raw_anime_data_path: str = os.path.join('artifacts','anime_data.csv')
    
@dataclass
class DataTransformationConfig:
    rating_df_pivot_path: str = os.path.join('artifacts','rating_df_pivot.csv')
    filtered_anime_list_path: str = os.path.join('artifacts','filtered_anime_list.csv')
    csr_data_path: str = os.path.join('artifacts','csr_data')
    
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    