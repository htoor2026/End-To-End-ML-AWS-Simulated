from pipelines.training_pipeline import ml_pipeline
from steps.dynamic_importer import dynamic_importer
from steps.prediction_service_loader import prediction_service_loader
from steps.predictor import predictor
from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step


MLFLOW_MODEL_NAME = "model"


@pipeline(enable_cache=False)
def continuous_deployment_pipeline():
    """Run a training job and deploy an MLflow model deployment."""

    # Run the training pipeline
    trained_model = ml_pipeline()

    # (Re)deploy the trained model
    mlflow_model_deployer_step(
        model=trained_model,
        deploy_decision=True,
        model_name=MLFLOW_MODEL_NAME,
        workers=3,
    )


@pipeline(enable_cache=False)
def inference_pipeline():
    """Run a batch inference job with data loaded from an API."""

    # Load batch data for inference
    batch_data = dynamic_importer()

    # Load the deployed model service
    model_deployment_service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        step_name="mlflow_model_deployer_step",
    )

    # Run predictions on the batch data
    predictor(
        service=model_deployment_service,
        input_data=batch_data,
    )