data "aws_iam_policy_document" "ml_s3_access" {
  statement {
    sid = "ListMLBucket"

    actions = [
      "s3:ListBucket"
    ]

    resources = [
      aws_s3_bucket.ml_data.arn
    ]
  }

  statement {
    sid = "ReadWriteMLObjects"

    actions = [
      "s3:GetObject",
      "s3:PutObject"
    ]

    resources = [
      "${aws_s3_bucket.ml_data.arn}/datasets/*",
      "${aws_s3_bucket.ml_data.arn}/models/*"
    ]
  }
}

resource "aws_iam_policy" "ml_s3_access" {
  name        = "${var.project_name}-s3-access"
  description = "Least-privilege S3 access for datasets and model artifacts."

  policy = data.aws_iam_policy_document.ml_s3_access.json
}

data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    actions = [
      "sts:AssumeRole"
    ]

    principals {
      type = "Service"

      identifiers = [
        "ec2.amazonaws.com"
      ]
    }
  }
}

resource "aws_iam_role" "ml_api_role" {
  name = "${var.project_name}-api-role"

  assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json
}

resource "aws_iam_role_policy_attachment" "ml_s3_access" {
  role       = aws_iam_role.ml_api_role.name
  policy_arn = aws_iam_policy.ml_s3_access.arn
}

resource "aws_iam_instance_profile" "ml_api" {
  name = "${var.project_name}-api-profile"
  role = aws_iam_role.ml_api_role.name
}