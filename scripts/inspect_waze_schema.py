"""
Script to inspect and summarize the schema of Parquet files in a given directory.
"""
import os
import sys
import glob
import pandas as pd
from collections import defaultdict

def inspect_parquet_schemas(directory_path):
    # Find all Parquet files in the target directory
    parquet_files = glob.glob(os.path.join(directory_path, '*.parquet'))
    
    # Exit early if no Parquet files are present
    if not parquet_files:
        print(f"No parquet files found in {directory_path}")
        return
    
    # Display the list of discovered Parquet files
    print(f"Found {len(parquet_files)} parquet files:")
    for file_path in parquet_files:
        # List each file name
        print(f"  - {os.path.basename(file_path)}")
    
    all_fields = defaultdict(list)
    
    print("\nFILE SCHEMAS:")
    print("="*80)
    
    # Iterate through each file to print detailed schema information
    for file_path in parquet_files:
        file_name = os.path.basename(file_path)
        print(f"\nFile: {file_name}")
        # Load data to inspect rows, columns, and types
        df = pd.read_parquet(file_path)
        
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        print("  Schema:")
        
        # Display each column name and its data type
        for col, dtype in df.dtypes.items():
            print(f"    - {col}: {dtype}")
            all_fields[col].append(file_name)
    
    print("\nCOMMON FIELDS ACROSS FILES:")
    print("="*80)
    
    # Summarize fields that are common across multiple files
    for field, files in all_fields.items():
        if len(files) > 1:
            print(f"\nField: {field}")
            print(f"  Appears in {len(files)} files:")
            for file in files:
                print(f"    - {file}")