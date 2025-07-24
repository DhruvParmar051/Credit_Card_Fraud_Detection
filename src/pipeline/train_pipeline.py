"""This module defines the pipeline execution for Credit Card Fraud Detection."""

import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils.logger import setup_logger
from src.utils.exception import CustomException

logger = setup_logger()


def run_pipeline():
    """
    Executes the full pipeline: Data Ingestion → Data Transformation → Model Training.
    """
    try:
        logger.info("🚀 Starting Credit Card Fraud Detection Pipeline...")

        logger.info("📥 Initiating Data Ingestion...")
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        logger.info("✅ Data Ingestion Complete: Train: %s, Test: %s", train_path, test_path)

        logger.info("🔄 Initiating Data Transformation...")
        transformation = DataTransformation()
        train_array, test_array, preprocessor_path = transformation.initiate_data_transformation(
            train_path, test_path
        )
        logger.info("✅ Data Transformation Complete. Preprocessor saved at: %s", preprocessor_path)

        logger.info("🤖 Initiating Model Training...")
        trainer = ModelTrainer()
        best_f1_score = trainer.initiate_model_trainer(train_array, test_array)
        logger.info("🏆 Best Model F1-score: %.4f", best_f1_score)

        logger.info("🎯 Pipeline Execution Completed Successfully!")

    except Exception as e:
        logger.error("❌ Pipeline Execution Failed: %s", str(e))
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    run_pipeline()
