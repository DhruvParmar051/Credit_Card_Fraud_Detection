"""Module to train and evaluate machine learning models."""

import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.utils.exception import CustomException
from src.utils.logger import setup_logger
from src.utils.save_object import save_object
from src.utils.evaluate_model import evaluate_model

os.environ["LOKY_MAX_CPU_COUNT"] = "6"
logger = setup_logger()


@dataclass
class ModelTrainerConfig:
    """Configuration for model trainer."""
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

# pylint: disable=too-few-public-methods
class ModelTrainer:
    """Class responsible for training and saving the best model."""

    def __init__(self):
        self.trained_model_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Trains multiple classification models, 
        selects the best model based on F1-score, 
        and saves it.
        """
        try:
            logger.info("📂 Splitting training and testing data...")

            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            logger.info("🚀 Initializing classification models...")

            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(),
                "CatBoost": CatBoostClassifier(verbose=False),
                "AdaBoost": AdaBoostClassifier(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
            }

            params = {
                "Logistic Regression": {"C": [0.01, 0.1, 1, 10]},
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 10, 20, 30],
                    "splitter": ["best", "random"],
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "criterion": ["gini", "entropy"],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.1, 0.3],
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 7],
                },
                "XGBoost": {
                    "learning_rate": [0.01, 0.1, 0.3],
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 7],
                },
                "CatBoost": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.5],
                },
                "K-Neighbors Classifier": {"n_neighbors": [3, 5, 7]},
            }

            logger.info("🔍 Evaluating models with hyperparameter tuning...")
            model_report = evaluate_model(x_train, y_train, x_test, y_test, models, params)

            best_model_name = max(model_report, key=lambda x: model_report[x]["f1_score"])
            best_model_score = model_report[best_model_name]["f1_score"]

            if best_model_score < 0.6:
                raise CustomException("❌ No suitable model found with an F1-score above 0.6", sys)

            logger.info(
                "🏆 Best model found: %s with F1-score: %.4f", best_model_name, 
                best_model_score
            )

            best_model = models[best_model_name]

            logger.info("💾 Saving the best model...")
            save_object(
                file_path=self.trained_model_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_score

        except Exception as e:
            logger.error("❌ Model training failed: %s", str(e))
            raise CustomException(e, sys) from e
