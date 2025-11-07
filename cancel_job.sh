#!/bin/bash

# Cancel specific job and all running jobs
REGION="ap-southeast-2"
JOB_QUEUE="solar-panel-inference-gpu-job-queue"
SPECIFIC_JOB_ID="47b137be-2fc6-40c6-a088-36db8b6e186c"

echo "Cancelling specific job: $SPECIFIC_JOB_ID"
aws batch cancel-job --region $REGION --job-id $SPECIFIC_JOB_ID --reason "Too many tiles - testing with subset" || \
aws batch terminate-job --region $REGION --job-id $SPECIFIC_JOB_ID --reason "Too many tiles - testing with subset"

echo "Cancelling all other jobs..."
for status in SUBMITTED RUNNABLE; do
  job_ids=$(aws batch list-jobs --region $REGION --job-queue $JOB_QUEUE --job-status $status --query 'jobList[].jobId' --output text)
  if [ -n "$job_ids" ] && [ "$job_ids" != "None" ]; then
    echo $job_ids | xargs -n1 -I {} aws batch cancel-job --region $REGION --job-id {} --reason "Too many tiles - testing with subset"
  fi
done

running_ids=$(aws batch list-jobs --region $REGION --job-queue $JOB_QUEUE --job-status RUNNING --query 'jobList[].jobId' --output text)
if [ -n "$running_ids" ] && [ "$running_ids" != "None" ]; then
  echo $running_ids | xargs -n1 -I {} aws batch terminate-job --region $REGION --job-id {} --reason "Too many tiles - testing with subset"
fi

echo "Done!"