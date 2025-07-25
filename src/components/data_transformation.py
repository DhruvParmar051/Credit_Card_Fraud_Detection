"""
This module contains classes and methods for data transformation,
including preprocessing pipelines for handling missing values and scaling.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from src.utils.logger import setup_logger
from src.utils.exception import CustomException
from src.utils.save_object import save_object

logger = setup_logger()


@dataclass
class DataTransformationConfig:
    """Configuration paths for data transformation artifacts."""
    preproceesor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    """Handles transformation of raw data into a machine-learning ready format."""

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_date_transformar_object(self):
        """
        Creates and returns a preprocessing pipeline that imputes missing values
        and scales numerical features.
        """
        try:
            numerical_columns = [f"V{i}" for i in range(1, 29)] + ["Amount"]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            logger.info("✅ Numerical data transformation pipeline created.")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns)
                ]
            )
            logger.info("✅ Data Preprocessor Created.")

            return preprocessor

        except Exception as e:
            logger.error("❌ Data Preprocessor Failed: %s", str(e))
            raise CustomException(e, sys) from e

    # pylint: disable=too-many-locals
    def initiate_data_transformation(self, train_path, test_path):
        """
        Applies transformation pipeline to training and testing datasets.

        Returns:
            Tuple: transformed train array, test array, and preprocessor path
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df = train_df.drop(columns=["Time"], errors="ignore").reset_index(drop=True)
            test_df = test_df.drop(columns=["Time"], errors="ignore").reset_index(drop=True)

            logger.info("🗑 Dropped 'Time' column and reset index.")
            logger.info("📂 Train and test data loaded successfully.")

            preprocessor = self.get_date_transformar_object()

            target_column_name = 'Class'

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logger.info("🔄 Applying preprocessing transformations...")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logger.info("💾 Saving preprocessing object...")

            save_object(
                file_path=self.data_transformation_config.preproceesor_obj_file_path,
                obj=preprocessor
            )

            logger.info("✅ Data Transformation Completed Successfully!")

            return (train_arr, test_arr, self.data_transformation_config.preproceesor_obj_file_path)

        except Exception as e:
            logger.error("❌ Data Transformation Failed: %s", str(e))
            raise CustomException(e, sys) from e
