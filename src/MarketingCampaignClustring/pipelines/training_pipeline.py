import os
import sys
from src.MarketingCampaignClustring.logger import logging
from src.MarketingCampaignClustring.exception import customexception

from src.MarketingCampaignClustring.components.data_ingestion import data_ingestion





obj1=data_ingestion()
train_data,test_data=obj1.initiate_data_ingestion()