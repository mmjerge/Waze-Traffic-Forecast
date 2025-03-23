"""
Model implementations for Waze traffic forecasting.
"""

from waze_traffic_forecast.models.stgformer import STGformer
from waze_traffic_forecast.models.layers import GraphPropagation, SpatiotemporalAttention