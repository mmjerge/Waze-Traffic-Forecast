"""
Waze Traffic Forecast

A package for traffic forecasting using graph transformer models on Waze data.
"""

__version__ = "0.1.0"

from waze_traffic_forecast.dataset import WazeGraphDataset
from waze_traffic_forecast._config import get_default_config