 # data Subpackage

 The `waze_traffic_forecast.data` subpackage provides modules for preprocessing raw Waze traffic data and constructing graph representations suitable for dynamic graph models.

 ## Modules

 - `graph_builder.py`: 
   - `WazeGraphBuilder` class
   - Methods to build node/edge DataFrames, create temporal graph snapshots, sample edges and subgraphs, and convert snapshots to PyTorch tensors.
 - `preprocessor.py`:  
   - Functions to clean, normalize, and merge raw segment and jam data files.
 - `inspector.py`:  
   - Utilities to inspect file schemas, data distributions, and field coverage across datasets.

 ## Workflow

 1. **Inspect Schema**: Use `inspector.py` to understand data fields and their types.
 2. **Preprocess Data**: Clean and normalize raw CSV/Parquet files via `WazePreprocessor` in `preprocessor.py`.
 3. **Build Graph**: Instantiate `WazeGraphBuilder` to extract nodes/edges and generate time-indexed snapshots.
 4. **Prepare Tensors**: Convert snapshots to feature (`X`) and adjacency (`A`) tensors for model training.