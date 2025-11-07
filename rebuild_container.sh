#!/bin/bash

cd infra/

echo "Authenticating with ECR..."
aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin 309379105600.dkr.ecr.ap-southeast-2.amazonaws.com

echo "Building container without cache..."
docker build --no-cache -t solar-inference .

echo "Tagging container..."
docker tag solar-inference:latest 309379105600.dkr.ecr.ap-southeast-2.amazonaws.com/solar-inference:latest

echo "Pushing to ECR..."
docker push 309379105600.dkr.ecr.ap-southeast-2.amazonaws.com/solar-inference:latest

echo "Verifying script in container..."
docker run --rm solar-inference:latest --help

echo "Done! Container rebuilt and pushed."