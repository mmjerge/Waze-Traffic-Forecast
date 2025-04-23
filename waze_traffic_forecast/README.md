 # waze_traffic_forecast Package

 The `waze_traffic_forecast` package contains the core components for processing Waze traffic data and training a graph transformer-based forecasting model.

 ## Subpackages and Modules

 - `_config.py`: Configuration loading, default settings, and YAML I/O functions.
 - `dataset.py`: `WazeGraphDataset` class for loading, processing, and batching traffic graph data.
 - `utils.py`: Utility functions and placeholders for future extensions.

 ### data
 Located in `waze_traffic_forecast/data/`
 - `graph_builder.py`: Builds graph structure and temporal snapshots from raw segment and jam data.
 - `preprocessor.py`: Cleans and preprocesses raw Waze data files for model input.
 - `inspector.py`: Inspects and summarizes data schemas for Parquet/CSV inputs.

 ### models
 Located in `waze_traffic_forecast/models/`
 - `graph_transformer.py`: Core graph transformer layer implementation.
 - `layers.py`: Definitions of custom neural network layers and normalization modules.
 - `stgformer.py`: `STGformerModel` class combining spatial-temporal attention and transformer blocks.

 ## Usage

 Import the package and use the provided classes to build datasets, configure models, and run training pipelines via the provided scripts.

 Example:
 ```python
 from waze_traffic_forecast._config import load_config
 from waze_traffic_forecast.dataset import WazeGraphDataset
 from waze_traffic_forecast.models.stgformer import STGformerModel

 config = load_config('config.yaml')
 dataset = WazeGraphDataset(data_dir=config['data']['directory'], **config['data'])
 model = STGformerModel(in_channels=dataset.X.shape[-1], ...)
 ```