#!/usr/bin/env python3
"""
tile_and_upload_to_s3.py

Create overlapping image tiles from input geo-JPEG images and upload them directly to S3.

Usage:
    python tile_and_upload_to_s3.py --input-dir images/ --s3-bucket solar-panel-inference-data --s3-prefix tiles/ --tile-size 1024 --overlap 256
"""

import argparse
import csv
import os
import tempfile
from io import BytesIO
from math import ceil
from pathlib import Path

import boto3
import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window
from tqdm import tqdm


def tile_and_upload_image(
    fp,
    s3_client,
    bucket,
    s3_prefix,
    tile_size=1024,
    overlap=256,
    out_format="jpg",
    csv_writer=None,
    max_tiles=None,
):
    """Tile a single image file and upload tiles directly to S3.

    Returns number of tiles uploaded.
    """
    uploaded = 0
    with rasterio.open(fp) as src:
        w, h = src.width, src.height
        step = tile_size - overlap
        nx = int(ceil((w - overlap) / step))
        ny = int(ceil((h - overlap) / step))

        base = os.path.splitext(os.path.basename(fp))[0]
        print(f"  Creating {nx}x{ny} = {nx*ny} tiles")

        for ix in range(nx):
            for iy in range(ny):
                x = ix * step
                y = iy * step
                win_w = min(tile_size, w - x)
                win_h = min(tile_size, h - y)
                win = Window(x, y, win_w, win_h)
                data = src.read(window=win)

                # Convert to HWC RGB for image writing
                if data.ndim == 3 and data.shape[0] >= 3:
                    img = np.dstack([data[0], data[1], data[2]])
                else:
                    # single band -> replicate
                    band = data[0] if data.ndim == 3 else data
                    img = np.stack([band, band, band], axis=-1)

                # Ensure uint8
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype("uint8")

                tile_name = f"{base}_{ix}_{iy}.{out_format}"
                s3_key = f"{s3_prefix.rstrip('/')}/{tile_name}"

                # Save to memory buffer
                buffer = BytesIO()
                Image.fromarray(img).save(
                    buffer, format="JPEG" if out_format == "jpg" else "PNG", quality=95
                )
                buffer.seek(0)

                # Upload to S3
                try:
                    s3_client.upload_fileobj(buffer, bucket, s3_key)
                    print(f"    Uploaded {tile_name}")
                except Exception as e:
                    print(f"    Failed to upload {tile_name}: {e}")
                    continue

                # Capture geotransform for this window
                transform = src.window_transform(win)
                crs = src.crs.to_string() if src.crs is not None else ""

                if csv_writer is not None:
                    row = {
                        "tile_name": tile_name,
                        "s3_key": s3_key,
                        "src_image": os.path.basename(fp),
                        "ix": ix,
                        "iy": iy,
                        "x_off": int(x),
                        "y_off": int(y),
                        "width": int(win_w),
                        "height": int(win_h),
                        "transform_a": transform.a,
                        "transform_b": transform.b,
                        "transform_c": transform.c,
                        "transform_d": transform.d,
                        "transform_e": transform.e,
                        "transform_f": transform.f,
                        "crs": crs,
                    }
                    csv_writer.writerow(row)

                uploaded += 1
                if max_tiles is not None and uploaded >= max_tiles:
                    return uploaded

    return uploaded


def main():
    parser = argparse.ArgumentParser(
        description="Tile geo-JPEG images and upload to S3"
    )
    parser.add_argument(
        "--input-dir", required=True, help="Directory with input geo-JPEG images"
    )
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket name")
    parser.add_argument(
        "--s3-prefix", default="tiles/", help="S3 prefix/folder for tiles"
    )
    parser.add_argument(
        "--tile-size", type=int, default=1024, help="Tile size in pixels"
    )
    parser.add_argument(
        "--overlap", type=int, default=256, help="Overlap between tiles in pixels"
    )
    parser.add_argument(
        "--format", choices=["jpg", "png"], default="jpg", dest="out_format"
    )
    parser.add_argument(
        "--csv", default="tiles_metadata.csv", help="Local CSV metadata file to write"
    )
    parser.add_argument(
        "--max-tiles-per-image",
        type=int,
        default=None,
        help="Limit tiles per source image (for testing)",
    )
    parser.add_argument("--region", default="ap-southeast-2", help="AWS region")
    args = parser.parse_args()

    # Initialize S3 client
    s3_client = boto3.client("s3", region_name=args.region)

    # Check if bucket exists
    try:
        s3_client.head_bucket(Bucket=args.s3_bucket)
        print(f"Using S3 bucket: s3://{args.s3_bucket}/{args.s3_prefix}")
    except Exception as e:
        print(f"Error accessing S3 bucket {args.s3_bucket}: {e}")
        return

    inp = Path(args.input_dir)
    if not inp.exists():
        print(f"Input directory {inp} does not exist")
        return

    # Find geo-JPEG files
    image_patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
    files = []
    for pat in image_patterns:
        files.extend(sorted(inp.glob(pat)))

    if not files:
        print(f"No geo-JPEG images found in {inp}")
        return

    print(f"Found {len(files)} geo-JPEG files to process")

    # CSV metadata setup
    fieldnames = [
        "tile_name",
        "s3_key",
        "src_image",
        "ix",
        "iy",
        "x_off",
        "y_off",
        "width",
        "height",
        "transform_a",
        "transform_b",
        "transform_c",
        "transform_d",
        "transform_e",
        "transform_f",
        "crs",
    ]

    with open(args.csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        total_uploaded = 0
        for fp in tqdm(files, desc="Processing images"):
            print(f"Tiling and uploading {fp.name}")
            try:
                uploaded = tile_and_upload_image(
                    str(fp),
                    s3_client,
                    args.s3_bucket,
                    args.s3_prefix,
                    tile_size=args.tile_size,
                    overlap=args.overlap,
                    out_format=args.out_format,
                    csv_writer=writer,
                    max_tiles=args.max_tiles_per_image,
                )
                print(f"  Uploaded {uploaded} tiles for {fp.name}")
                total_uploaded += uploaded
            except Exception as e:
                print(f"  Failed to process {fp}: {e}")

    print(
        f"Done! Uploaded {total_uploaded} tiles to s3://{args.s3_bucket}/{args.s3_prefix}"
    )
    print(f"Metadata saved to: {args.csv}")


if __name__ == "__main__":
    main()
