import os
import glob
import pandas as pd
import pyarrow.parquet as pq
from collections import defaultdict

def inspect_waze_schemas(directory_path):
    """
    Inspect Waze parquet files schemas and identify relationships.
    
    Args:
        directory_path (str): Path to the directory containing Waze parquet files
    """
    print(f"Inspecting Waze parquet files in: {directory_path}")
    
    parquet_files = glob.glob(os.path.join(directory_path, '*.parquet'))
    
    if not parquet_files:
        print(f"No parquet files found in {directory_path}")
        return
    
    print(f"Found {len(parquet_files)} parquet files.")
    
    for file_path in parquet_files:
        file_name = os.path.basename(file_path)
        print(f"\nFile: {file_name}")
        
        try:
            schema = pq.read_schema(file_path)
            print(f"  Fields ({len(schema.names)}):")
            
            for field in schema:
                print(f"    - {field.name}: {field.type}")
                
        except Exception as e:
            print(f"  Error reading schema: {str(e)}")
    
    return True