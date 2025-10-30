#!/bin/bash

# Set variables
JOB_NAME="solar-inference-job-$(date +%Y%m%d-%H%M%S)"
JOB_QUEUE="solar-panel-inference-gpu-job-queue"
JOB_DEFINITION="solar-panel-inference-gpu-job-definition"
REGION="ap-southeast-2"

# Submit the job
aws batch submit-job \
  --region $REGION \
  --job-name $JOB_NAME \
  --job-queue $JOB_QUEUE \
  --job-definition $JOB_DEFINITION \
  --parameters '{
    "s3Tiles": "s3://solar-panel-inference-data/tiles/",
    "outS3": "s3://solar-panel-inference-data/outputs/detections.geojson"
  }'

echo "Job submitted: $JOB_NAME"
echo "Monitor at: https://console.aws.amazon.com/batch/home?region=$REGION#jobs/queue/$JOB_QUEUE"