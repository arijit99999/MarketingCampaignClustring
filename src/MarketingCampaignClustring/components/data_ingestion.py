import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from src.MarketingCampaignClustring.logger import logging
from src.MarketingCampaignClustring.exception import customexception

class dataingestionConfig:
    raw_data=os.path.join("artifacts","raw.csv")
    train_data=os.path.join("artifacts","train.csv")
    test_data=os.path.join("artifacts","test.csv")


class data_ingestion:
    def __init__(self):
        self.ingestion_config=dataingestionConfig()
        
    
    def initiate_data_ingestion(self):
        logging.info("data ingestion started")
        
        try:
           data=pd.read_csv(r"C:\Users\deyar\OneDrive\Desktop\MarketingCampaignClustring\notebooks\data\model.csv") 
           os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data)),exist_ok=True) 
           u={"class":"classes"}
           data=data.rename(columns=u)
           data.to_csv(self.ingestion_config.raw_data,index=False) 
           logging.info('row data saved ')  
           train_data,test_data=train_test_split(data,test_size=.2,random_state=121)  
           logging.info('split train and test data')   

           train_data.to_csv(self.ingestion_config.train_data,index=False)
           logging.info("saved train data")
           test_data.to_csv(self.ingestion_config.test_data,index=False)
           logging.info('saved test data')
           return (self.ingestion_config.train_data,self.ingestion_config.test_data)
        except Exception as e:
           logging.info("exception during occured at data ingestion stage")
           raise customexception(e,sys)