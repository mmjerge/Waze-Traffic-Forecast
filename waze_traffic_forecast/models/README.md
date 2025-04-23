 # models Subpackage

 The `waze_traffic_forecast.models` subpackage contains the implementation of the STGformer model and supporting layers for spatio‑temporal traffic forecasting.

 ## Modules

 - `layers.py`: Definitions of custom neural network layers, including spatial propagation, attention, and normalization layers.
 - `graph_transformer.py`: Implementation of the core graph transformer block for spatial message passing and temporal encoding.
 - `stgformer.py`: `STGformerModel` class that stacks multiple graph transformer layers, handles input/output processing, and defines the forward pass for prediction.

 ## Model Architecture

 1. **Graph Propagation Layers**: Capture spatial dependencies between road segments or junctions.
 2. **Spatio-Temporal Attention**: Combine local and global temporal patterns across snapshots.
 3. **Positional Encoding**: Encode time-step information to preserve sequence ordering.

 The `STGformerModel` can be customized via parameters such as number of layers, hidden channels, number of attention heads, and dropout rates.