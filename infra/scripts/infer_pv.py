"""
yolo_infer_geotiff.py
Runs YOLOv8 inference on tiled GeoTIFFs, converts detections to georeferenced GeoJSON,
and writes results to a GeoJSON file and PostGIS table.

Usage:
    python yolo_infer_geotiff.py --tiles_dir tiles/ --model best.pt --out detections.geojson
"""

import argparse
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import psycopg2
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform_bounds
from shapely.geometry import box, mapping
from tqdm import tqdm
from ultralytics import YOLO  # pip install ultralytics

# ---- Helpers ----


def pixel_to_world(transform: Affine, px: float, py: float):
    """Convert pixel coords (col, row) to world (x,y) using rasterio Affine."""
    x, y = transform * (px, py)
    return x, y


def bbox_pixels_to_world(transform: Affine, bbox_px):
    """bbox_px = (x_min, y_min, x_max, y_max) in pixel coords relative to the tile."""
    xmin_px, ymin_px, xmax_px, ymax_px = bbox_px
    x_min, y_min = pixel_to_world(transform, xmin_px, ymin_px)
    x_max, y_max = pixel_to_world(transform, xmax_px, ymax_px)
    return (
        x_min,
        y_max,
        x_max,
        y_min,
    )  # return as xmin,ymin,xmax,ymax in world coords (y may flip)


def read_geotiff_tiles(tiles_dir):
    """Yield (tile_path, numpy_image (H,W,C), transform, crs)"""
    p = Path(tiles_dir)
    tif_paths = list(p.glob("*.tif")) + list(p.glob("*.tiff"))
    if not tif_paths:
        raise FileNotFoundError("No GeoTIFFs found in tiles_dir")
    for tif in tif_paths:
        with rasterio.open(tif) as src:
            # read as HWC uint8
            arr = src.read()  # (bands, rows, cols)
            # convert to HWC and normalize if needed
            if arr.shape[0] >= 3:
                img = np.dstack([arr[0], arr[1], arr[2]])
            else:
                img = arr[0]
                img = np.stack([img, img, img], axis=-1)
            yield str(tif), img, src.transform, src.crs


# ---- Main pipeline ----


def run_inference_on_tiles(
    tiles_dir, model_path, conf_thresh=0.35, iou=0.45, out_geojson="detections.geojson"
):
    model = YOLO(model_path)  # load model
    features = []

    for tif_path, img, transform, crs in tqdm(
        list(read_geotiff_tiles(tiles_dir)), desc="Tiles"
    ):
        # ultralytics expects images as numpy HWC BGR or file path; we pass numpy RGB
        # If image is large, consider resizing or using tile size previously set.
        # Run inference
        results = model.predict(
            source=img,
            imgsz=1024,
            conf=conf_thresh,
            iou=iou,
            max_det=1000,
            device=0,
            verbose=False,
        )
        # results is a list; single element for our image
        res = results[0]
        # res.boxes.xyxy: tensor Nx4 in pixel coords relative to input image
        # However when passing numpy, ultralytics may have resized the image — using results.orig_shape and results.boxes.xyxy to map back.
        # Use res.boxes.xyxy.to('cpu').numpy() and res.boxes.conf, res.boxes.cls
        boxes = (
            res.boxes.xyxy.cpu().numpy()
        )  # (N,4): x1,y1,x2,y2 in model input pixel space
        confs = res.boxes.conf.cpu().numpy()
        classes = (
            res.boxes.cls.cpu().numpy().astype(int)
            if hasattr(res.boxes, "cls")
            else np.zeros(len(confs), dtype=int)
        )

        # Need mapping from model input coords to original tile pixel coords:
        # ultralytics provides res.orig_shape and res.ori_image_shape? Use res.orig_shape and res.ori_shape if available.
        # Simpler approach: run model.predict with 'save=False' and set imgsz equal to tile size to avoid resizing.
        # For robustness, verify shapes:
        ih, iw = img.shape[0], img.shape[1]
        # If boxes coordinates are in same size, we can map directly:
        # Clip boxes to image bounds
        for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, classes):
            # ensure coords are within image
            x1 = max(0, min(iw, float(x1)))
            x2 = max(0, min(iw, float(x2)))
            y1 = max(0, min(ih, float(y1)))
            y2 = max(0, min(ih, float(y2)))
            # Convert to geospatial bbox (world coords)
            # Note: rasterio's pixel coordinate origin is top-left; pixel_to_world handles that
            # compute world coords for four corners
            xmin_w, ymax_w = pixel_to_world(
                transform, x1, y2
            )  # bottom-left pixel -> world
            xmax_w, ymin_w = pixel_to_world(
                transform, x2, y1
            )  # top-right pixel -> world
            # Construct polygon (xmin,ymin,xmax,ymax)
            geom = box(xmin_w, ymin_w, xmax_w, ymax_w)
            prop = {
                "confidence": float(conf),
                "class": int(cls),
                "tile": os.path.basename(tif_path),
            }
            features.append(
                {"type": "Feature", "geometry": mapping(geom), "properties": prop}
            )

    # Build GeoDataFrame and save
    if not features:
        print("No detections found.")
        return None

    gdf = gpd.GeoDataFrame.from_features(features, crs=crs.to_string())
    # Optionally convert CRS to EPSG:4326 for PostGIS (if your PostGIS expects 4326)
    try:
        gdf = gdf.to_crs(epsg=4326)
    except Exception:
        pass
    gdf.to_file(out_geojson, driver="GeoJSON")
    print(f"Wrote {len(gdf)} detections to {out_geojson}")
    return out_geojson


# ---- PostGIS uploader ----
def upload_geojson_to_postgis(geojson_path, pg_conn_str, table_name="detections"):
    # pg_conn_str e.g. "host=... dbname=... user=... password=... port=5432"
    conn = psycopg2.connect(pg_conn_str)
    cur = conn.cursor()
    # create table if not exists
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

    # load geojson
    gdf = gpd.read_file(geojson_path)
    # ensure geometry is 4326
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
        "--tiles_dir", required=True, help="Directory with GeoTIFF tiles"
    )
    parser.add_argument("--model", required=True, help="Path to YOLOv8 model .pt")
    parser.add_argument(
        "--out", default="detections.geojson", help="Output GeoJSON path"
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IOU threshold")
    parser.add_argument(
        "--upload_pg", action="store_true", help="Upload results to PostGIS"
    )
    parser.add_argument("--pg_conn", default=None, help="PostGIS connection string")
    parser.add_argument("--pg_table", default="detections", help="PostGIS table name")
    args = parser.parse_args()

    geojson = run_inference_on_tiles(
        args.tiles_dir,
        args.model,
        conf_thresh=args.conf,
        iou=args.iou,
        out_geojson=args.out,
    )
    if args.upload_pg:
        if not args.pg_conn:
            raise ValueError("pg_conn is required when --upload_pg is set")
        upload_geojson_to_postgis(geojson, args.pg_conn, table_name=args.pg_table)


if __name__ == "__main__":
    main()
