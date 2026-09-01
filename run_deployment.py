import click

from pipelines.deployment_pipeline import continuous_deployment_pipeline
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)


PIPELINE_NAME = "continuous_deployment_pipeline"
DEPLOYMENT_STEP_NAME = "mlflow_model_deployer_step"

# IMPORTANT:
# This is the MLflow artifact/service name.
# It is NOT the ZenML model name "prices_predictor".
MLFLOW_MODEL_NAME = "model"


def delete_existing_services():
    """Delete old MLflow services created by this deployment pipeline."""

    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # Find every service created by this pipeline/step.
    # Do not filter by model name here because old code may
    # have created services with inconsistent names.
    existing_services = model_deployer.find_model_server(
        pipeline_name=PIPELINE_NAME,
        pipeline_step_name=DEPLOYMENT_STEP_NAME,
    )

    if not existing_services:
        print(
            "[yellow]No previous MLflow deployment services found.[/yellow]"
        )
        return

    print(
        f"[yellow]Found {len(existing_services)} old deployment "
        f"service(s). Removing them...[/yellow]"
    )

    for service in existing_services:
        try:
            print(
                f"Deleting service: {service.uuid}"
            )

            try:
                print(
                    f"Model name: {service.config.model_name}"
                )
            except Exception:
                pass

            try:
                print(
                    f"Model URI: {service.config.model_uri}"
                )
            except Exception:
                pass

            model_deployer.delete_model_server(
                service.uuid,
                timeout=20,
                force=True,
            )

            print(
                f"[green]Deleted service {service.uuid}[/green]"
            )

        except Exception as error:
            print(
                f"[red]Could not delete service "
                f"{service.uuid}: {error}[/red]"
            )


def show_running_service():
    """Display the newly deployed MLflow prediction service."""

    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    services = model_deployer.find_model_server(
        pipeline_name=PIPELINE_NAME,
        pipeline_step_name=DEPLOYMENT_STEP_NAME,
        model_name=MLFLOW_MODEL_NAME,
        running=True,
    )

    if not services:
        print(
            "[red]No running MLflow service was found.[/red]"
        )
        return

    print(
        f"[green]Found {len(services)} running "
        f"MLflow service(s).[/green]"
    )

    for service in services:
        print(
            "\n[bold green]MLflow prediction service[/bold green]"
        )

        print(
            f"Service UUID: {service.uuid}"
        )

        print(
            f"Model name: {service.config.model_name}"
        )

        print(
            f"Model URI: {service.config.model_uri}"
        )

        print(
            f"Prediction URL: {service.prediction_url}"
        )


@click.command()
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="Delete existing MLflow prediction services",
)
def run_main(stop_service: bool):
    """Run the prices predictor deployment pipeline."""

    if stop_service:
        delete_existing_services()
        return

    print(
        "[bold yellow]"
        "Removing previous deployment services..."
        "[/bold yellow]"
    )

    delete_existing_services()

    print(
        "[bold cyan]"
        "Starting fresh continuous deployment..."
        "[/bold cyan]"
    )

    # Train and deploy a fresh model
    continuous_deployment_pipeline()

    print(
        "[bold green]"
        "Continuous deployment pipeline completed."
        "[/bold green]"
    )

    show_running_service()

    print(
        "\nMLflow UI:\n"
        f"mlflow ui --backend-store-uri '{get_tracking_uri()}'"
    )


if __name__ == "__main__":
    run_main()