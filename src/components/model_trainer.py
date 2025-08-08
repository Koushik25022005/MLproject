import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, 
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainingConfig:
    trained_model_path=os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        super().__init__()
        self.model_trainer = ModelTrainingConfig()
        
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split into train and test input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            models = {
                "Random Forest": RandomForestRegressor(), 
                "Decison Tree": DecisionTreeRegressor(), 
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGB Classifier": XGBRegressor(),
                "CatBoost Classifier": CatBoostRegressor(verbose=False),
                "K-Neighbors Classifiers": KNeighborsRegressor(),
                "Adaboost Classifier": AdaBoostRegressor()
            }
            
            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test = y_test ,models=models)
            
            # To get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))
            
            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                 list(model_report.values()).index(best_model_score)
                ]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.75: 
                raise CustomException("No best model found")
            
            logging.info("Best model found on both training and testing datasets")
            best_model.fit(X_train, y_train)
            save_object(
                file_path=self.model_trainer.trained_model_path,
                obj=best_model
            )   
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            
            return r2_square
        except Exception as e:
            raise CustomException(e, sys)
        
        

    