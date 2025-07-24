import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.utils.logger import setup_logger
from src.utils.exception import CustomException

logger = setup_logger() 

@dataclass 
class DataIngestionConfig:
    
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        
        logger.info("📥 Starting Data Ingestion Process...")
        
        try:
            
            df = pd.read_csv('notebook/data/creditcard.csv')
            logger.info(f"✅ Successfully loaded dataset. Shape: {df.shape}")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logger.info(f"✅ Raw data saved at {self.ingestion_config.raw_data_path}")

            logger.info("🔀 Splitting dataset into train and test sets...")
            
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logger.info(f"✅ Train data saved at {self.ingestion_config.train_data_path} (Size: {train_set.shape})")
            
            logger.info(f"✅ Test data saved at {self.ingestion_config.test_data_path} (Size: {test_set.shape})")
            
            logger.info("📥 Data Ingestion Completed Successfully! 🎯")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            
            logger.error(f"❌ Data Ingestion Failed: {str(e)}")
            
            raise CustomException(e, sys)
