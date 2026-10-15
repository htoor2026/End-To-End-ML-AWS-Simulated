resource "aws_cloudwatch_log_group" "ml_api" {
  name              = "/ml/${var.project_name}/api"
  retention_in_days = 7

  tags = {
    Project = var.project_name
  }
}


resource "aws_cloudwatch_metric_alarm" "high_cpu" {
  alarm_name        = "${var.project_name}-high-cpu"
  alarm_description = "Alerts when ML API EC2 CPU usage remains high."

  namespace   = "AWS/EC2"
  metric_name = "CPUUtilization"

  statistic = "Average"
  period    = 300

  evaluation_periods = 2

  comparison_operator = "GreaterThanThreshold"
  threshold           = 80

  dimensions = {
    InstanceId = aws_instance.ml_api.id
  }
}