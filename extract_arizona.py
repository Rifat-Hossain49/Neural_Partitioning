import os
import shapefile
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

def extract_bboxes(directory, output_csv, sample_size=None):
    """
    Reads all .shp files in the directory and extracts bounding boxes to a CSV.
    Format: min_0, min_1, max_0, max_1 (equivalent to min_x, min_y, max_x, max_y)
    """
    all_bboxes = []
    shp_files = [f for f in os.listdir(directory) if f.endswith('.shp')]
    
    print(f"Found {len(shp_files)} shapefiles. Starting extraction...")
    
    total_records = 0
    for shp_file in shp_files:
        path = os.path.join(directory, shp_file)
        print(f"Processing {shp_file}...")
        try:
            with shapefile.Reader(path) as sf:
                # Iterate through shapes and extract bounding box
                for shape in tqdm(sf.shapes(), desc=f"Reading {shp_file}", leave=False):
                    if hasattr(shape, 'bbox') and len(shape.bbox) == 4:
                        bbox = shape.bbox # [min_x, min_y, max_x, max_y]
                    elif hasattr(shape, 'points') and len(shape.points) > 0:
                        # Fallback for points or shapes without bbox
                        pts = np.array(shape.points)
                        min_x, min_y = pts.min(axis=0)
                        max_x, max_y = pts.max(axis=0)
                        bbox = [min_x, min_y, max_x, max_y]
                    else:
                        continue

                    all_bboxes.append({
                        'min_0': bbox[0],
                        'min_1': bbox[1],
                        'max_0': bbox[2],
                        'max_1': bbox[3]
                    })
                    total_records += 1
        except Exception as e:
            print(f"Error reading {shp_file}: {e}")
            
    if not all_bboxes:
        print("No data extracted.")
        return 0

    df = pd.DataFrame(all_bboxes)
    
    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size} records from {len(df)} total...")
        df = df.sample(n=sample_size, random_state=42)
        total_records = len(df)
    elif sample_size:
        print(f"Requested sample size {sample_size} is >= total records {len(df)}. Keeping all.")

    print(f"Saving {total_records} records to {output_csv}...")
    df.to_csv(output_csv, index=False)
    print("Done!")
    return total_records

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract bounding boxes from shapefiles.")
    parser.add_argument("--dir", type=str, default="Arizona", help="Directory containing shapefiles")
    parser.add_argument("--output", type=str, default="arizona_extracted.csv", help="Output CSV file")
    parser.add_argument("--sample", type=int, default=None, help="Number of rows to sample after extraction")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"Directory '{args.dir}' not found.")
    else:
        count = extract_bboxes(args.dir, args.output, args.sample)
        print(f"\nFinal row count in {args.output}: {count}")
