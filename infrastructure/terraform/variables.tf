variable "aws_region" {
  description = "AWS region used for the educational architecture."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Name prefix for simulated AWS-ready infrastructure."
  type        = string
  default     = "house-price-ml"
}
variable "vpc_id" {
  description = "VPC ID for a future AWS deployment."
  type        = string
}

variable "ami_id" {
  description = "AMI ID for a future EC2 instance."
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type for the ML API."
  type        = string
  default     = "t3.micro"
}

variable "allowed_api_cidr" {
  description = "CIDR allowed to access the FastAPI service."
  type        = string
}