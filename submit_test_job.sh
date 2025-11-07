#!/bin/bash

# Test job with limited tiles
JOB_NAME="solar-inference-test-$(date +%Y%m%d-%H%M%S)"
JOB_QUEUE="solar-panel-inference-gpu-job-queue"
JOB_DEFINITION="solar-panel-inference-gpu-job-definition"
REGION="ap-southeast-2"

# Submit job with specific tile subset (first 10 tiles only)
aws batch submit-job \
  --region $REGION \
  --job-name $JOB_NAME \
  --job-queue $JOB_QUEUE \
  --job-definition $JOB_DEFINITION \
  --parameters '{
    "s3Tiles": "s3://solar-panel-inference-data/images/",
    "outS3": "s3://solar-panel-inference-data/outputs/test-detections.geojson",
    "conf": "0.01",
    "maxTiles": "20"
  }'

echo "Test job submitted: $JOB_NAME"
echo "Monitor at: https://console.aws.amazon.com/batch/home?region=$REGION#jobs/queue/$JOB_QUEUE"