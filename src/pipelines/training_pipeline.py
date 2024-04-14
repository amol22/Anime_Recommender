import os
import sys
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer

if __name__=="__main__":
    try:
        obj = DataIngestion()
        anime_path,rating_path = obj.data_ingestion()
        data_transform = DataTransformation()
        train_data = data_transform.initiate_data_transformation(anime_path,rating_path)
        model = ModelTrainer()
        model.initiate_model_trainer(train_data)
    except Exception as e:
        raise CustomException(e,sys)