import logging
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------
# Abstract Base Class for Model Evaluation Strategy
# ---------------------------------------------------------
class ModelEvaluationStrategy(ABC):
    """
    Abstract strategy for evaluating machine learning models.

    Any concrete evaluation strategy must implement
    the evaluate_model() method.
    """

    @abstractmethod
    def evaluate_model(
        self,
        model: RegressorMixin,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:
        """
        Evaluate a trained model.

        Parameters
        ----------
        model : RegressorMixin
            Trained regression model.

        X_test : pd.DataFrame
            Test feature dataset.

        y_test : pd.Series
            True target values.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics.
        """
        pass


# ---------------------------------------------------------
# Regression Evaluation Strategy
# ---------------------------------------------------------
class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
    """
    Evaluation strategy for regression models.

    Calculates:
    - Mean Absolute Error
    - Mean Squared Error
    - Root Mean Squared Error
    - R-Squared
    """

    def evaluate_model(
        self,
        model: RegressorMixin,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:

        logging.info("Generating predictions on test data.")

        y_pred = model.predict(X_test)

        logging.info("Calculating regression evaluation metrics.")

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse**0.5
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "MAE": float(mae),
            "MSE": float(mse),
            "RMSE": float(rmse),
            "R2": float(r2),
        }

        logging.info("Model evaluation completed.")
        logging.info("MAE: %.4f", mae)
        logging.info("MSE: %.4f", mse)
        logging.info("RMSE: %.4f", rmse)
        logging.info("R2: %.4f", r2)

        return metrics


# ---------------------------------------------------------
# Context Class
# ---------------------------------------------------------
class ModelEvaluator:
    """
    Context class that executes a selected
    model evaluation strategy.

    This makes it possible to swap evaluation strategies
    without changing the rest of the ML pipeline.
    """

    def __init__(self, strategy: ModelEvaluationStrategy):
        self._strategy = strategy

    def set_strategy(
        self,
        strategy: ModelEvaluationStrategy,
    ) -> None:
        """
        Change the evaluation strategy.

        Parameters
        ----------
        strategy : ModelEvaluationStrategy
            New evaluation strategy.
        """

        logging.info("Switching model evaluation strategy.")

        self._strategy = strategy

    def evaluate(
        self,
        model: RegressorMixin,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:
        """
        Evaluate the model using the currently
        selected strategy.

        Parameters
        ----------
        model : RegressorMixin
            Trained regression model.

        X_test : pd.DataFrame
            Test features.

        y_test : pd.Series
            True target values.

        Returns
        -------
        dict
            Evaluation metrics.
        """

        logging.info(
            "Evaluating model using %s.",
            self._strategy.__class__.__name__,
        )

        return self._strategy.evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
        )


# ---------------------------------------------------------
# Example / Manual Testing
# ---------------------------------------------------------
if __name__ == "__main__":
    # This file is normally called by the ZenML
    # model evaluation step.
    #
    # Do not create fake variables such as:
    #
    # model = trained_sklearn_model
    # X_test = test_data_features
    # y_test = test_data_target
    #
    # unless you actually define/load them here.

    pass