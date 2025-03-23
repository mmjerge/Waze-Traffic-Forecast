import os
import sys
import glob
import pandas as pd
from collections import defaultdict

def inspect_parquet_schemas(directory_path):
    parquet_files = glob.glob(os.path.join(directory_path, '*.parquet'))
    
    if not parquet_files:
        print(f"No parquet files found in {directory_path}")
        return
    
    print(f"Found {len(parquet_files)} parquet files:")
    for file_path in parquet_files:
        print(f"  - {os.path.basename(file_path)}")
    
    all_fields = defaultdict(list)
    
    print("\nFILE SCHEMAS:")
    print("="*80)
    
    for file_path in parquet_files:
        file_name = os.path.basename(file_path)
        print(f"\nFile: {file_name}")
        
        df = pd.read_parquet(file_path)
        
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        print("  Schema:")
        
        for col, dtype in df.dtypes.items():
            print(f"    - {col}: {dtype}")
            all_fields[col].append(file_name)
    
    print("\nCOMMON FIELDS ACROSS FILES:")
    print("="*80)
    
    for field, files in all_fields.items():
        if len(files) > 1:
            print(f"\nField: {field}")
            print(f"  Appears in {len(files)} files:")
            for file in files:
                print(f"    - {file}")