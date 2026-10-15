resource "aws_security_group" "ml_api" {
  name        = "${var.project_name}-api-sg"
  description = "Security group for the ML prediction API."
  vpc_id      = var.vpc_id

  ingress {
    description = "FastAPI access"

    from_port = 8001
    to_port   = 8001
    protocol  = "tcp"

    cidr_blocks = [
      var.allowed_api_cidr
    ]
  }

  egress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"

    cidr_blocks = [
      "0.0.0.0/0"
    ]
  }

  tags = {
    Project = var.project_name
  }
}

resource "aws_instance" "ml_api" {
  ami           = var.ami_id
  instance_type = var.instance_type

  iam_instance_profile = (
    aws_iam_instance_profile.ml_api.name
  )

  vpc_security_group_ids = [
    aws_security_group.ml_api.id
  ]

  tags = {
    Name    = "${var.project_name}-api"
    Project = var.project_name
  }
}