# Solar Panel Inference Pipeline

This project provides an AWS Batch-based pipeline for detecting solar panels in aerial imagery using Roboflow Inference.

## Overview

The pipeline consists of three main steps:
1. **Tile** large geo-JPEG images into smaller, overlapping tiles
2. **Upload** tiles to S3 for processing
3. **Run** batch inference jobs to detect solar panels
4. **Collect** georeferenced detection results as GeoJSON

## Prerequisites

- AWS CLI configured with appropriate permissions
- Python 3.8+ with required packages
- Roboflow account with a trained solar panel detection model

## Setup

### 1. Install Dependencies

```bash
pip install rasterio pillow boto3 tqdm numpy geopandas shapely
```

### 2. Configure Roboflow Credentials

Store your Roboflow credentials securely in AWS Systems Manager:

```bash
aws ssm put-parameter \
  --name "/roboflow/model_id" \
  --value "your-workspace/your-model-version" \
  --type "String" \
  --region ap-southeast-2

aws ssm put-parameter \
  --name "/roboflow/api_key" \
  --value "your-roboflow-api-key" \
  --type "SecureString" \
  --region ap-southeast-2
```

### 3. Deploy Infrastructure

```bash
cd infra/
terraform init
terraform apply
```

## Usage

### Step 1: Tile and Upload Images

Convert your large geo-JPEG aerial images into smaller tiles and upload them to S3:

```bash
python tile_and_upload_to_s3.py \
  --input-dir "path/to/your/aerial/images/" \
  --s3-bucket "solar-panel-inference-data" \
  --s3-prefix "tiles/" \
  --tile-size 1024 \
  --overlap 256
```

**Parameters:**
- `--input-dir`: Directory containing your geo-JPEG files
- `--s3-bucket`: S3 bucket name (created by Terraform)
- `--s3-prefix`: Folder path in S3 for tiles
- `--tile-size`: Size of each tile in pixels (default: 1024)
- `--overlap`: Overlap between adjacent tiles in pixels (default: 256)

**Example output:**
```
Found 5 geo-JPEG files to process
Tiling and uploading image1.jpg
  Creating 4x3 = 12 tiles
  Uploaded 12 tiles for image1.jpg
Done! Uploaded 60 tiles to s3://solar-panel-inference-data/tiles/
```

### Step 2: Submit Inference Job

Run the batch job to process all tiles:

```bash
./submit_inference_job.sh
```

This script will:
- Submit a job to AWS Batch
- Process all tiles in `s3://solar-panel-inference-data/tiles/`
- Output results to `s3://solar-panel-inference-data/outputs/detections.geojson`

**Monitor job progress:**
- Check the AWS Batch console: [Job Queues](https://console.aws.amazon.com/batch/home?region=ap-southeast-2#jobs)
- View logs in CloudWatch: [Log Groups](https://console.aws.amazon.com/cloudwatch/home?region=ap-southeast-2#logsV2:log-groups)

### Step 3: Download Results

Download the detection results:

```bash
aws s3 cp s3://solar-panel-inference-data/outputs/detections.geojson ./results.geojson
```

The GeoJSON file contains:
- Solar panel detections as polygons
- Confidence scores for each detection
- Source tile information
- Coordinates in WGS84 (EPSG:4326)

## File Structure

```
solar-pv-inferencing/
├── README.md                          # This file
├── tile_and_upload_to_s3.py          # Tiling and upload script
├── submit_inference_job.sh           # Job submission script
├── infra/                            # Terraform infrastructure
│   ├── batch.tf                      # AWS Batch configuration
│   ├── s3.tf                         # S3 bucket setup
│   ├── iam.tf                        # IAM roles and policies
│   ├── cloudwatch.tf                 # Logging configuration
│   ├── variables.tf                  # Terraform variables
│   └── scripts/
│       └── infer_pv_s3.py            # Inference script (runs in container)
└── Dockerfile                        # Container definition
```

## Configuration

### Tiling Parameters

- **Tile Size**: 1024x1024 pixels works well for most aerial imagery
- **Overlap**: 256 pixels (25%) ensures detections aren't missed at tile boundaries
- **Format**: JPEG provides good compression for aerial imagery

### Batch Job Parameters

Edit `submit_inference_job.sh` to customize:
- Input S3 path (`s3Tiles` parameter)
- Output S3 path (`outS3` parameter)
- Confidence threshold (modify the inference script)

### Instance Types

The infrastructure supports multiple GPU instance types:
- `g4dn.xlarge` - Most cost-effective
- `g5.xlarge` - Good performance/cost balance  
- `g6.xlarge` - Latest generation (most expensive)

AWS Batch will automatically select available spot instances.

## Troubleshooting

### Common Issues

**1. "No tiles found" error**
- Ensure your images are in geo-JPEG format with embedded geospatial information
- Check that files have `.jpg` or `.jpeg` extensions

**2. Job stuck in "SUBMITTED" state**
- Check spot instance availability in your region
- Verify IAM permissions are correctly configured

**3. "Missing Roboflow credentials" error**
- Ensure SSM parameters are set correctly
- Check parameter names match exactly: `/roboflow/model_id` and `/roboflow/api_key`

**4. Out of memory errors**
- Reduce tile size (try 512x512)
- Ensure your images aren't corrupted or unusually large

### Monitoring

- **Job Status**: AWS Batch Console → Job Queues
- **Logs**: CloudWatch → Log Groups → `/aws/batch/solar-panel-inference`
- **Costs**: AWS Cost Explorer (filter by Batch service)

## Cost Optimization

- Use spot instances (already configured)
- Process images in batches to minimize cold starts
- Delete intermediate tiles after processing
- Set S3 lifecycle rules to automatically delete old results

## Support

For issues with:
- **Infrastructure**: Check Terraform documentation
- **Roboflow Integration**: Consult Roboflow Inference documentation
- **AWS Services**: Review AWS Batch and S3 documentation