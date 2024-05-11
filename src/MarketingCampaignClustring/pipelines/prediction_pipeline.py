import os
import sys
import pickle
import pandas as pd
import numpy as np
from src.MarketingCampaignClustring.logger import logging
from src.MarketingCampaignClustring.exception import customexception
from src.MarketingCampaignClustring.utlis.utils import load_object

class model_pred_config:
    preprcessor_path=os.path.join('artifacts',"preprocessor.pkl")
    model_path=os.path.join('artifacts',"model.pkl")
class model_prediction:
    def __init__(self):
        self.model_pred=model_pred_config()
    def model_pred_initiate(self,features):
        try:
            preprocessor=load_object(self.model_pred.preprcessor_path)
            model=load_object(self.model_pred.model_path)
            scaled_data=preprocessor.transform(features)
            pred=model.predict(scaled_data)
            return pred
        except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise customexception(e,sys)

class custom_data:
    def __init__(self,invoice_frequency,total_quantity,total_bill,day_gap,country):
        self.invoice_frequency=invoice_frequency
        self.total_quantity=total_quantity
        self.total_bill=total_bill
        self.day_gap=day_gap
        self.country=country

    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    'invoice_frequency':[self.invoice_frequency],
                    'total_quantity':[self.total_quantity],
                    'total_bill':[self.total_bill],
                    'day_gap':[self.day_gap],
                    'country':[self.country]}
                df = pd.DataFrame(custom_data_input_dict)
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise customexception(e,sys)
    