#!/usr/bin/env python3
"""
Local test script to validate the inference pipeline without AWS
"""
import os
import sys
import tempfile
from pathlib import Path

# Add the scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "infra" / "scripts"))

def test_script_imports():
    """Test that all imports work"""
    print("Testing imports...")
    try:
        import argparse
        import json
        import os
        import sys
        import tempfile
        from pathlib import Path
        import boto3
        import geopandas as gpd
        import numpy as np
        import psycopg2
        import rasterio
        from inference import get_model
        try:
            from inference_sdk import InferenceHTTPClient
            print("✓ InferenceHTTPClient available")
        except ImportError:
            print("⚠ InferenceHTTPClient not available (workflow support disabled)")
        from shapely.geometry import box, mapping
        from tqdm import tqdm
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_argument_parsing():
    """Test that argument parsing works"""
    print("\nTesting argument parsing...")
    try:
        from infer_pv_s3 import main
        # Test help output
        import subprocess
        result = subprocess.run([
            sys.executable, "infra/scripts/infer_pv_s3.py", "--help"
        ], capture_output=True, text=True)
        
        if "--max_tiles" in result.stdout:
            print("✓ --max_tiles argument found")
            return True
        else:
            print("✗ --max_tiles argument missing")
            print("Help output:", result.stdout)
            return False
    except Exception as e:
        print(f"✗ Argument parsing error: {e}")
        return False

def test_container_locally():
    """Test the container locally"""
    print("\nTesting container locally...")
    try:
        import subprocess
        result = subprocess.run([
            "docker", "run", "--rm", "solar-inference:latest", "--help"
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and "--max_tiles" in result.stdout:
            print("✓ Container test successful")
            print("Available arguments:")
            for line in result.stdout.split('\n'):
                if '--' in line:
                    print(f"  {line.strip()}")
            return True
        else:
            print("✗ Container test failed")
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)
            return False
    except Exception as e:
        print(f"✗ Container test error: {e}")
        return False

def main():
    print("=== Local Testing ===")
    
    tests = [
        test_script_imports,
        test_argument_parsing,
        test_container_locally
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n=== Results ===")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All tests passed! Ready to deploy.")
    else:
        print("✗ Some tests failed. Fix issues before deploying.")
        sys.exit(1)

if __name__ == "__main__":
    main()