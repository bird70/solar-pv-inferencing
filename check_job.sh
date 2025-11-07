#!/bin/bash

REGION="ap-southeast-2"
JOB_QUEUE="solar-panel-inference-gpu-job-queue"

echo "=== Current Jobs ==="
aws batch list-jobs --region $REGION --job-queue $JOB_QUEUE --job-status RUNNING --query 'jobList[].[jobName,jobId,jobStatus]' --output table

echo -e "\n=== Running EC2 Instances ==="
aws ec2 describe-instances --region $REGION --filters "Name=instance-state-name,Values=running" --query 'Reservations[].Instances[].[InstanceId,InstanceType,LaunchTime]' --output table

echo -e "\n=== Latest Job Details ==="
JOB_ID=$(aws batch list-jobs --region $REGION --job-queue $JOB_QUEUE --job-status RUNNING --query 'jobList[0].jobId' --output text)
if [ "$JOB_ID" != "None" ] && [ -n "$JOB_ID" ]; then
    aws batch describe-jobs --region $REGION --jobs $JOB_ID --query 'jobs[0].attempts[0].taskProperties.{InstanceType:instanceType,InstanceId:instanceId}' --output table
fi