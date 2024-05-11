import pandas as pd
import numpy as np
import os
import sys
import mlflow
from urllib.parse import urlparse

from src.MarketingCampaignClustring.logger import logging
from src.MarketingCampaignClustring.exception import customexception
from src.MarketingCampaignClustring.utlis.utils import load_object
from sklearn.metrics import accuracy_score,precision_recall_fscore_support


class modelevaluation:
    pass
    def modelevaluationint(self,train_arr,test_arr):
        k=os.path.join('artifacts','model.pkl')
        x_train=train_arr[:,:-1]
        y_train=train_arr[:,-1:]
        x_test=test_arr[:,:-1]
        y_test=test_arr[:,-1:]
        if len(y_train.shape) > 1 and y_train.shape[1] == 1:
             y_train= y_train.ravel()
        if len(y_test.shape) > 1 and y_test.shape[1] == 1:
             y_test= y_test.ravel()
        mlflow.set_registry_uri("https://dagshub.com/arijit99999/MarketingCampaignClustring.mlflow")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print(tracking_url_type_store)

        with mlflow.start_run():                        
          model=load_object(k)
          p=model.predict(x_test)
          score=accuracy_score(y_test,p)*100
          mlflow.log_metric("accuracy",score)
          mlflow.log_param("model","adaboost")
          if tracking_url_type_store != "file":
              mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
          else:
              mlflow.sklearn.log_model(model,"model")


     