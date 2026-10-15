resource "aws_s3_bucket" "ml_data" {
  bucket = "${var.project_name}-data-local"

  tags = {
    Project     = var.project_name
    Environment = "local-simulation"
    ManagedBy   = "terraform"
  }
}