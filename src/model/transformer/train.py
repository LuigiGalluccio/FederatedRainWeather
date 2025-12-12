import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yaml
import os
import logging as log
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === PATHS ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
from logger import Logger
from dataset import Dataset
from model import myTransformer


def load_config(config_path="config.yaml"):
    """Carica il file di configurazione YAML."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_data(config):

    csv_path = config['data']['csv_path']
    input_window = config['data']['input_window']
    output_window = config['data']['output_window']
    feature_cols = config['data']['feature_cols']
    target_cols = config['data']['target_cols']
    batch_size = config['training']['batch_size']

    # Dataset completo per stimare la lunghezza totale
    full_dataset = Dataset(csv_path, input_window, output_window, feature_cols, target_cols)
    total_len = len(full_dataset.data)

    train_end = int(0.8 * total_len)
    val_end = int(0.9 * total_len)

    # 80-10-10
    train_dataset = Dataset(
        csv_path, input_window, output_window,
        feature_cols, target_cols,
        start_idx=0, end_idx=train_end
    )
    val_dataset = Dataset(
        csv_path, input_window, output_window,
        feature_cols, target_cols,
        start_idx=train_end, end_idx=val_end
    )
    test_dataset = Dataset(
        csv_path, input_window, output_window,
        feature_cols, target_cols,
        start_idx=val_end
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def load_data_client(config, place_id):

    csv_path = config['data']['csv_path']
    input_window = config['data']['input_window']
    output_window = config['data']['output_window']
    feature_cols = config['data']['feature_cols']
    target_cols = config['data']['target_cols']
    batch_size = config['training']['batch_size']

    df = pd.read_csv(csv_path, parse_dates=['datetime'])
    df_place = df[df['station_id'] == place_id].sort_values('datetime')

    log.info(f"[CLIENT {place_id}] -> LOADED {len(df_place)} ROWS FOR STATION_ID={place_id}")

    total_len = len(df_place)
    train_end = int(0.7 * total_len)
    val_end = int(0.85 * total_len)

    train_dataset = Dataset(
        df_place, input_window, output_window,
        feature_cols, target_cols,
        start_idx=0, end_idx=train_end
    )
    val_dataset = Dataset(
        df_place, input_window, output_window,
        feature_cols, target_cols,
        start_idx=train_end, end_idx=val_end
    )
    test_dataset = Dataset(
        df_place, input_window, output_window,
        feature_cols, target_cols,
        start_idx=val_end
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def build_model(config, device):
  
    feature_cols = config['data']['feature_cols']
    target_cols = config['data']['target_cols']

    model = myTransformer(
        input_dim=len(feature_cols),
        output_dim=len(target_cols)
    ).to(device)

    return model


def train(model, train_loader, criterion, optimizer, device, output_window, feature_dim):

    model.train()
    epoch_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        # Decoder input: shape [B, T_out, feature_dim]
        tgt_in = torch.zeros((x.size(0), output_window, feature_dim), device=x.device)

        pred = model(x, tgt_in)
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

            tgt_in = torch.zeros((x.size(0), output_window, feature_dim), device=x.device)
            pred = model(x, tgt_in)

            loss = criterion(pred, y)
            val_loss += loss.item()

    return val_loss / len(val_loader)


def evaluate(model, test_loader, device, output_window, feature_dim, target_cols):

    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            tgt_in = torch.zeros((x.size(0), output_window, feature_dim), device=x.device)
            pred = model(x, tgt_in)

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
    config = load_config()
    log = Logger()

    lr = config['training']['learning_rate']
    epochs = config['training']['epochs']
    device = torch.device(config['training']['device'])
    batch_size = config['training']['batch_size']

    feature_cols = config['data']['feature_cols']
    target_cols = config['data']['target_cols']
    output_window = config['data']['output_window']

    feature_dim = len(feature_cols)

    log.info("TRANSFORMER HYPER-PARAMETERS:")
    log.info(f'Learning rate: {lr}')
    log.info(f'Epochs: {epochs}')
    log.info(f'Device: {device}')
    log.info(f'Batch size: {batch_size}')
    log.info(f'Feature cols ({len(feature_cols)}): {feature_cols}')
    log.info(f'Target cols ({len(target_cols)}): {target_cols}\n')

    # Data
    train_loader, val_loader, test_loader = load_data(config)

    model = build_model(config, device)

    # Loss & optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

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
