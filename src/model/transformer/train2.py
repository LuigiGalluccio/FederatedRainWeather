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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
from logger import Logger
from dataset import Dataset
from model import myTransformer

def load_config(config_path="config.yaml"):
    """load YAML config file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

df_10min = load_weather_data(base_path="../../../data/storage/vantage-pro-ws1/2025", downsample_factor=60)


def load_data(config):

    input_window = config['data']['input_window']
    output_window = config['data']['output_window']
    feature_cols = config['data']['feature_cols']
    target_cols = config['data']['target_cols']
    batch_size = config['training']['batch_size']

    print(input_window, output_window, feature_cols, target_cols)

    dataset = WeatherDataset(
        df=df_10min,
        input_window=input_window,
        output_window=output_window,
        feature_cols=feature_cols,
        target_cols=target_cols
    )   

    # print(len(dataset))
    total_len = len(dataset)

    train_end = int(0.8 * total_len)
    val_end = int(0.9 * total_len)

    train_dataset = WeatherDataset(
        df=df_10min,
        input_window=input_window,
        output_window=output_window,
        feature_cols=feature_cols,
        target_cols=target_cols,
        start_idx=0,
        end_idx=train_end
    )
    val_dataset = WeatherDataset(
        df=df_10min,
        input_window=input_window,
        output_window=output_window,
        feature_cols=feature_cols,
        target_cols=target_cols,
        start_idx=train_end, 
        end_idx=val_end
    )
    test_dataset = WeatherDataset(
        df=df_10min,
        input_window=input_window,
        output_window=output_window,
        feature_cols=feature_cols,
        target_cols=target_cols,
        start_idx=val_end
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def build_model(config, device):
  
    feature_cols = config['data']['feature_cols']
    target_cols = config['data']['target_cols']

    model = myTransformer(
        input_dim=len(feature_cols),
        output_dim=len(target_cols),
        input_window=config['data']['input_window'],
        output_window=config['data']['output_window']
    ).to(device)

    return model



def train(model, train_loader, criterion, optimizer, device, output_window, feature_dim):

    model.train()
    epoch_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        # Decoder input: shape [B, T_out, feature_dim]

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)

def validate(model, val_loader, criterion, device, output_window, feature_dim):

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)

            loss = criterion(pred, y)
            val_loss += loss.item()

    return val_loss / len(val_loader)


def evaluate(model, test_loader, device, output_window, feature_dim, target_cols):

    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)      # [B, T_out, n_targets]
    all_targets = np.concatenate(all_targets, axis=0)  # [B, T_out, n_targets]

    results = []
    for i, target_name in enumerate(target_cols):
        preds_flat = all_preds[:, :, i].reshape(-1)
        targets_flat = all_targets[:, :, i].reshape(-1)

        mae = mean_absolute_error(targets_flat, preds_flat)
        rmse = mean_squared_error(targets_flat, preds_flat) ** 0.5
        r2score_val = r2_score(targets_flat, preds_flat)

        results.append((target_name, mae, rmse, r2score_val))

    return results


def main():     
    config = load_config("config.yaml")
    log = Logger()

    lr = config['training']['learning_rate']
    epochs = config['training']['epochs']
    device = torch.device(config['training']['device'])
    batch_size = config['training']['batch_size']
    input_window = config['data']['input_window']
    output_window = config['data']['output_window']
    feature_cols = config['data']['feature_cols']
    target_cols = config['data']['target_cols']
    train_loader, val_loader, test_loader = load_data(config)
    feature_dim = len(feature_cols)
    model = build_model(config, device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        
    log.info("TRANSFORMER HYPER-PARAMETERS:")
    log.info(f'Learning rate: {lr}')
    log.info(f'Epochs: {epochs}')
    log.info(f'Device: {device}')
    log.info(f'Batch size: {batch_size}')
    log.info(f'Feature cols ({len(feature_cols)}): {feature_cols}')
    log.info(f'Target cols ({len(target_cols)}): {target_cols}\n')

        # Training loop
    for epoch in range(epochs):
        train_loss = train(
            model, train_loader, criterion, optimizer,
            device, output_window, feature_dim
        )
        val_loss = validate(
            model, val_loader, criterion,
            device, output_window, feature_dim
        )

        log.info(
            f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

    results = evaluate(
        model, test_loader, device,
        output_window, feature_dim, target_cols
    )

    log.info("\nValutazione finale:")
    for name, mae, rmse, r2 in results:
        log.info(f"MAE ({name}): {mae:.4f}")
        log.info(f"RMSE ({name}): {rmse:.4f}")
        log.info(f"R2 SCORE ({name}): {r2:.4f}")



if __name__ == "__main__":
    main()