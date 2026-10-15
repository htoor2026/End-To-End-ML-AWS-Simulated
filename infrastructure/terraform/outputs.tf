output "data_bucket_name" {
  description = "Name of the S3 bucket for ML datasets."
  value       = aws_s3_bucket.ml_data.bucket
}