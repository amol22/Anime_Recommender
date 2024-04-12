import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.config import ModelTrainerConfig
from src.utils import save_object

from sklearn.neighbors import NearestNeighbors

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig
        
    def initiate_model_trainer(self,csr_data):
        logging.info("Model training initiated")
        try:
            knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=12, n_jobs=-1)
            knn.fit(csr_data)
            
            logging.info("Model training complete")
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=knn)
            
        except Exception as e:
            raise CustomException(e,sys)