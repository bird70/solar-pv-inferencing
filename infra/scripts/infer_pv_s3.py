#!/usr/bin/env python3
"""
s3_yolo_infer.py

- Downloads or streams GeoTIFF tiles from S3 (supports s3://bucket/prefix/*.tif)
- Downloads YOLOv8 model from S3 to local path (or can use local .pt)
- Runs YOLOv8 inference on each tile (supports bbox and mask)
- Converts detections to georeferenced GeoJSON (EPSG:4326)
- Writes output GeoJSON to local file and uploads to S3 (s3://bucket/output/prefix/detections.geojson)
- Optionally uploads to PostGIS

Requires:
pip install ultralytics rasterio geopandas shapely boto3 tqdm psycopg2-binary fsspec s3fs

Usage example:
python s3_yolo_infer.py --s3_tiles s3://my-bucket/tiles/ --model_s3 s3://my-bucket/models/best.pt \
    --out_s3 s3://my-bucket/outputs/auckland_detections.geojson --tmp_dir /tmp/run1 --conf 0.35
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import boto3
import geopandas as gpd
import numpy as np

# Optional PostGIS uploader
import psycopg2
import rasterio

# from ultralytics import YOLO
from inference import get_model
from shapely.geometry import box, mapping
from tqdm import tqdm

# ---- Helpers ----


def list_s3_objects(s3_uri, s3_client):
    # expects s3_uri like s3://bucket/prefix/
    assert s3_uri.startswith("s3://")
    _, rest = s3_uri.split("s3://", 1)
    bucket, *key_parts = rest.split("/", 1)
    prefix = key_parts[0] if key_parts else ""
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    keys = []
    for p in pages:
        for obj in p.get("Contents", []):
            keys.append(f"s3://{bucket}/{obj['Key']}")
    return keys


def download_s3_file(s3_uri, local_path, s3_client):
    assert s3_uri.startswith("s3://")
    _, rest = s3_uri.split("s3://", 1)
    bucket, key = rest.split("/", 1)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_client.download_file(bucket, key, local_path)
    return local_path


def s3_path_parts(s3_uri):
    _, rest = s3_uri.split("s3://", 1)
    bucket, *key = rest.split("/", 1)
    key = key[0] if key else ""
    return bucket, key


def upload_file_to_s3(local_path, s3_uri, s3_client):
    bucket, key = s3_path_parts(s3_uri)
    s3_client.upload_file(local_path, bucket, key)
    return True


# Pixel -> world
def pixel_to_world(transform, px, py):
    x, y = transform * (px, py)
    return x, y


# Convert box pixels (x1,y1,x2,y2) to world bbox (xmin,ymin,xmax,ymax)
def bbox_pixels_to_world(transform, bbox_px):
    x1, y1, x2, y2 = bbox_px
    xmin, ymin = pixel_to_world(transform, x1, y1)
    xmax, ymax = pixel_to_world(transform, x2, y2)
    # rasterio uses top-left origin; ensure ymin < ymax by swapping if needed
    xmin_w, ymin_w = min(xmin, xmax), min(ymin, ymax)
    xmax_w, ymax_w = max(xmin, xmax), max(ymin, ymax)
    return xmin_w, ymin_w, xmax_w, ymax_w


# ---- Core pipeline ----


def process_tile_with_model(tile_path, model, conf_thr=0.35, iou=0.45, use_masks=False):
    """
    Run model.predict on the tile image (tile_path can be a local path).
    Returns list of dicts: {geometry: shapely geom, confidence: float, class: int}
    """
    # read image with rasterio to get transform and crs
    with rasterio.open(tile_path) as src:
        arr = src.read()
        # build HWC RGB
        if arr.shape[0] >= 3:
            img = np.dstack([arr[0], arr[1], arr[2]])
        else:
            img = np.stack([arr[0]] * 3, axis=-1)
        transform = src.transform
        crs = src.crs

    # Run Roboflow inference
    results = model.infer(img, confidence=conf_thr)
    outputs = []
    
    # Process Roboflow results format
    for prediction in results.get('predictions', []):
        x = prediction['x']
        y = prediction['y'] 
        width = prediction['width']
        height = prediction['height']
        confidence = prediction['confidence']
        class_id = prediction.get('class_id', 0)
        
        # Convert center+width/height to x1,y1,x2,y2
        x1 = x - width/2
        y1 = y - height/2
        x2 = x + width/2
        y2 = y + height/2
        
        xmin_w, ymin_w, xmax_w, ymax_w = bbox_pixels_to_world(
            transform, (x1, y1, x2, y2)
        )
        geom = box(xmin_w, ymin_w, xmax_w, ymax_w)
        outputs.append(
            {"geometry": geom, "confidence": float(confidence), "class": int(class_id)}
        )
    
    return outputs, crs


def run_s3_pipeline(
    s3_tiles_prefix,
    out_s3_uri,
    tmp_dir="/tmp/run",
    conf=0.35,
    iou=0.45,
    use_masks=False,
    upload_pg=False,
    pg_conn=None,
    pg_table="detections",
):
    os.makedirs(tmp_dir, exist_ok=True)
    s3 = boto3.client("s3")

    # Get Roboflow credentials from AWS Systems Manager
    ssm = boto3.client('ssm')
    
    try:
        model_id = ssm.get_parameter(Name='/roboflow/model_id', WithDecryption=True)['Parameter']['Value']
        api_key = ssm.get_parameter(Name='/roboflow/api_key', WithDecryption=True)['Parameter']['Value']
    except Exception as e:
        # Fallback to environment variables
        model_id = os.environ.get('ROBOFLOW_MODEL_ID')
        api_key = os.environ.get('ROBOFLOW_API_KEY')
        if not model_id or not api_key:
            raise ValueError("Missing Roboflow credentials. Set SSM parameters or environment variables.")
    
    print(f"Loading Roboflow model: {model_id}")
    model = get_model(model_id=model_id, api_key=api_key)

    # list tiles
    tile_uris = list_s3_objects(s3_tiles_prefix, s3)
    # filter for .tif/.tiff
    tile_uris = [u for u in tile_uris if u.lower().endswith((".tif", ".tiff"))]
    print(f"Found {len(tile_uris)} tiles")

    all_features = []
    out_crs = None

    for tile_uri in tqdm(tile_uris, desc="tiles"):
        # stream-read via rasterio's vsis3 path if AWS creds available in environment
        # vsis3 path: /vsis3/bucket/key
        bucket, key = s3_path_parts(tile_uri)
        vsis3_path = f"/vsis3/{bucket}/{key}"
        try:
            # Use rasterio to open vsis3 path directly (requires AWS creds or IAM role)
            with rasterio.open(vsis3_path) as src:
                local_tile = os.path.join(tmp_dir, os.path.basename(key))
                # We will still save a local copy to ensure ultralytics handling is robust
                src.close()
                s3.download_file(bucket, key, local_tile)
        except Exception:
            # fallback: download
            local_tile = os.path.join(tmp_dir, os.path.basename(key))
            download_s3_file(tile_uri, local_tile, s3)

        detections, crs = process_tile_with_model(
            local_tile, model, conf_thr=conf, iou=iou, use_masks=use_masks
        )
        if not out_crs:
            out_crs = crs
        for d in detections:
            geom = d["geometry"]
            feat = {
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {
                    "confidence": d.get("confidence"),
                    "class": d.get("class"),
                    "tile": os.path.basename(key),
                },
            }
            all_features.append(feat)
        # remove local tile to save disk
        try:
            os.remove(local_tile)
        except Exception:
            pass

    if not all_features:
        print("No detections produced.")
        return None

    gdf = gpd.GeoDataFrame.from_features(
        all_features, crs=out_crs.to_string() if out_crs is not None else "EPSG:3857"
    )
    # convert to 4326 for portability
    try:
        gdf = gdf.to_crs(epsg=4326)
    except Exception:
        pass

    local_out = os.path.join(tmp_dir, os.path.basename(out_s3_uri))
    gdf.to_file(local_out, driver="GeoJSON")
    print("Wrote local GeoJSON:", local_out)

    # upload to S3
    upload_file_to_s3(local_out, out_s3_uri, s3)
    print("Uploaded GeoJSON to:", out_s3_uri)

    # optional PostGIS upload
    if upload_pg:
        if not pg_conn:
            raise ValueError("pg_conn required for PostGIS upload")
        upload_geojson_to_postgis_local(local_out, pg_conn, pg_table)

    return out_s3_uri


# Simple PostGIS uploader using local file (re-use previous small uploader)
def upload_geojson_to_postgis_local(geojson_path, pg_conn_str, table_name="detections"):
    conn = psycopg2.connect(pg_conn_str)
    cur = conn.cursor()
    cur.execute(
        f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY,
        geom GEOMETRY(Geometry, 4326),
        confidence REAL,
        class INT,
        tile TEXT
    );
    """
    )
    conn.commit()
    gdf = gpd.read_file(geojson_path)
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    for _, row in gdf.iterrows():
        geom_wkt = row.geometry.wkt
        conf = (
            row.get("confidence", None)
            or row.get("properties", {}).get("confidence", None)
            or 0.0
        )
        cls = int(row.get("class", 0))
        tile = row.get("tile", None)
        cur.execute(
            f"INSERT INTO {table_name} (geom, confidence, class, tile) VALUES (ST_SetSRID(ST_GeomFromText(%s), 4326), %s, %s, %s);",
            (geom_wkt, float(conf), cls, tile),
        )
    conn.commit()
    cur.close()
    conn.close()
    print(f"Uploaded {len(gdf)} rows to PostGIS table '{table_name}'.")


# ---- CLI ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--s3_tiles",
        required=True,
        help="S3 prefix with tiles, e.g., s3://bucket/tiles/",
    )

    parser.add_argument(
        "--out_s3",
        required=True,
        help="S3 uri for output geojson, e.g., s3://bucket/outputs/detections.geojson",
    )
    parser.add_argument("--tmp_dir", default="/tmp/run", help="Local temp dir")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--use_masks", action="store_true")
    parser.add_argument("--upload_pg", action="store_true")
    parser.add_argument("--pg_conn", default=None)
    parser.add_argument("--pg_table", default="detections")
    args = parser.parse_args()

    run_s3_pipeline(
        args.s3_tiles,
        args.out_s3,
        tmp_dir=args.tmp_dir,
        conf=args.conf,
        iou=args.iou,
        use_masks=args.use_masks,
        upload_pg=args.upload_pg,
        pg_conn=args.pg_conn,
        pg_table=args.pg_table,
    )


if __name__ == "__main__":
    main()
