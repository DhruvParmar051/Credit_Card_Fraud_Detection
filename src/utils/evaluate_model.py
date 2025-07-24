# evaluate_model.py
# pylint: disable=invalid-name, too-many-arguments, too-many-positional-arguments, too-many-locals

"""Evaluate multiple classification models and log their performance metrics."""

import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV

from src.utils.logger import setup_logger
from src.utils.exception import CustomException

logger = setup_logger()

# pylint: disable=too-many-arguments
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluates classification models using accuracy, precision, recall, and F1-score.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data
        models (dict): Dictionary of classification models
        params (dict): Dictionary of hyperparameter grids for each model

    Returns:
        dict: Dictionary containing model evaluation metrics
    """
    try:
        report = {}

        for model_name, model in models.items():
            logger.info("🚀 Training %s...", model_name)

            best_params = {}

            if model_name in params and params[model_name]:
                search_iter = min(10, len(params[model_name]))
                rs = RandomizedSearchCV(
                    model,
                    params[model_name],
                    cv=3,
                    scoring="f1",
                    n_jobs=-1,
                    n_iter=search_iter,
                    random_state=42
                )

                rs.fit(X_train, y_train)
                best_params = rs.best_params_
                model.set_params(**best_params)
                logger.info("✅ Best parameters for %s: %s", model_name, best_params)

            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred)
            recall = recall_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred)

            report[model_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "params": best_params
            }

            logger.info(
                "📊 %s - Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1-score: %.4f",
                model_name, accuracy, precision, recall, f1
            )

        return report

    except Exception as e:
        logger.error("❌ Model evaluation failed: %s", str(e))
        raise CustomException(e, sys) from e
