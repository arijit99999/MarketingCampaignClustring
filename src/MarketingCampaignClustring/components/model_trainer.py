
import pandas as pd
import numpy as np
import os
import sys
from src.MarketingCampaignClustring.logger import logging
from src.MarketingCampaignClustring.exception import customexception
from src.MarketingCampaignClustring.utlis.utils import save_object
from sklearn.metrics import accuracy_score,precision_recall_fscore_support

from sklearn.ensemble import AdaBoostRegressor

 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,train_arr,test_arr):
        try:
         
            x_train=train_arr[:,:-1]
            y_train=train_arr[:,-1:]
            x_test=test_arr[:,:-1]
            y_test=test_arr[:,-1:]
            if len(y_train.shape) > 1 and y_train.shape[1] == 1:
             y_train= y_train.ravel()
            if len(y_test.shape) > 1 and y_test.shape[1] == 1:
             y_test= y_test.ravel()
            logging.info('read x_train,x_test,y_train,y_test')
            model=AdaBoostRegressor()
            model.fit(x_train,y_train)
            y_pred=model.predict(x_test)
            score=accuracy_score(y_test,y_pred)
            logging.info(f'model score is {score}')
            print(round(score*100,2),"%")
            print(precision_recall_fscore_support(y_test,y_pred))
            save_object(obj=model,file_path=self.model_trainer_config.trained_model_file_path)
            logging.info('model traing has been complited and we have saved the model')
            return self.model_trainer_config.trained_model_file_path
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise customexception(e,sys)