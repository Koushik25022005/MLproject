import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from  sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
import os


class DataTransformConfig:
    preprocessor_obj_file_path = os.path.join('artfacts', "preprocessors.pkl")
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformatino_config=DataTransformConfig()
        
    def get_data_transformer(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            cat_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
                 
                ]
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(statergy="median")),
                    ("scaler", StandardScaler),
                ]
                
                
            )
            logging.info("Numerical cloumns are encoded successsfully")
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(statergy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler())
                ]
                
            )
            logging.info("Categorical columns are encoded successfully")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, cat_columns)
                ]
            )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test datasets done")
            
            logging.info("Obtaining preprocessor object")
            
            preprocessor_obj = self.get_data_transformer()
            
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            input_feature_test_df = test_df[target_column_name]
            
            logging.info("""
                         Applying preprocessing object on training and testing datasets
                         """)
            
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transofrm(input_feature_test_df)
            
            
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
                         