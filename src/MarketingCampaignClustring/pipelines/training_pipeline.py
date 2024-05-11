import os
import sys
from src.MarketingCampaignClustring.logger import logging
from src.MarketingCampaignClustring.exception import customexception

from src.MarketingCampaignClustring.components.data_ingestion import data_ingestion
from src.MarketingCampaignClustring.components.data_transformation import data_transformation
from src.MarketingCampaignClustring.components.model_trainer import ModelTrainer




obj1=data_ingestion()
train_data,test_data=obj1.initiate_data_ingestion()

obj2=data_transformation()
train_arr,test_arr=obj2.data_transform_initiated(train_data,test_data)

obj3=ModelTrainer()
model=obj3.initate_model_training(train_arr,test_arr)