import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import os
import sys
import pandas as pd
import logging as log
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import glob
from dataset2 import WeatherDataset, load_weather_data
from model import myTransformer

df_10min = load_weather_data(base_path="storage/vantage-pro/2025", downsample_factor=60)

dataset = WeatherDataset(
    df=df_10min,
    input_window=6,
    output_window=6,
    feature_cols = ["Barometer","TempIn","HumIn","TempOut","WindSpeed","WindDir","HumOut"],
    target_col="RainRate",
)   

print(dataset[0])