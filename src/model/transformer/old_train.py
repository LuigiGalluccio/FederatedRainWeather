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
import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import personalizzati
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
from logger import Logger
from dataset import load_weather_data, preprocess_weather_data, WeatherDataset
from model import myTransformer

def load_config(config_path="config.yaml"):
    """Carica il file di configurazione YAML."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def get_global_preprocessed_df(config):
    """
    Carica tutte le stazioni specificate, le unisce e applica 
    il preprocessing globale (Seno/Coseno + Normalizzazione).
    """
    stations = [1, 3, 4, 5]
    dfs = []
    
    for s in stations:
        # Nota: assicurati che il path sia corretto rispetto alla posizione di train.py
        path = f"../../../data/storage/vantage-pro-ws{s}/2025"
        try:
            df_s = load_weather_data(base_path=path, downsample_factor=60, station_id=s)
            dfs.append(df_s)
            print(f"Loaded station {s}, shape: {df_s.shape}")
        except Exception as e:
            print(f"Errore nel caricamento della stazione {s}: {e}")

    if not dfs:
        raise ValueError("Nessun dato caricato. Controlla i path delle stazioni.")

    df_raw = pd.concat(dfs, axis=0).reset_index(drop=True)
    
    # Applichiamo preprocessing globale (WindDir -> Sin/Cos e StandardScaler)
    df_normalized, scaler = preprocess_weather_data(df_raw)
    return df_normalized, scaler

def load_data_client(config, place_id, df_full=None):
    """
    Carica i loader per un singolo client (stazione) partendo dal DF globale.
    """
    if df_full is None:
        df_full, _ = get_global_preprocessed_df(config)

    input_window = config['data']['input_window']
    output_window = config['data']['output_window']
    feature_cols = config['data']['feature_cols']
    target_cols = config['data']['target_cols']
    batch_size = config['training']['batch_size']

    # Filtro per stazione specifica
    df_place = df_full[df_full['station_id'] == place_id].sort_values('Datetime').reset_index(drop=True)

    total_len = len(df_place)
    total_len = len(df_place)
    print(f"DEBUG: Stazione {place_id} ha {total_len} righe totali dopo il preprocessing.")
    if total_len == 0:
        print(f"ERRORE: La stazione {place_id} è VUOTA! Forse non ci sono dati di pioggia?")
        # Gestisci l'errore o salta la stazione
    train_end = int(0.7 * total_len)
    val_end = int(0.85 * total_len)

    train_ds = WeatherDataset(df_place, input_window, output_window, feature_cols, target_cols, 0, train_end)
    val_ds = WeatherDataset(df_place, input_window, output_window, feature_cols, target_cols, train_end, val_end)
    test_ds = WeatherDataset(df_place, input_window, output_window, feature_cols, target_cols, val_end)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

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

def evaluate(model, test_loader, device, output_window, feature_cols, target_cols):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    results = []
    for i, target_name in enumerate(target_cols):
        preds_flat = all_preds[:, :, i].reshape(-1)
        targets_flat = all_targets[:, :, i].reshape(-1)
        mae = mean_absolute_error(targets_flat, preds_flat)
        rmse = np.sqrt(mean_squared_error(targets_flat, preds_flat))
        r2 = r2_score(targets_flat, preds_flat)
        results.append((target_name, mae, rmse, r2))
    return results

def main():     
    config = load_config("config.yaml")
    logger = Logger()

    # Caricamento centralizzato per test
    df_10min, scaler = get_global_preprocessed_df(config)
    
    device = torch.device(config['training']['device'])
    feature_cols = config['data']['feature_cols']
    target_cols = config['data']['target_cols']
    
    # Per il training centralizzato usiamo una stazione come esempio o tutte
    # Qui usiamo la logica client per caricare i dati della stazione 1
    train_loader, val_loader, test_loader = load_data_client(config, place_id=1, df_full=df_10min)
    
    model = build_model(config, device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=1e-4)

    logger.info("Starting Centralized Training (Station 1 Example)...")
    for epoch in range(config['training']['epochs']):
        t_loss = train(model, train_loader, criterion, optimizer, device, config['data']['output_window'], len(feature_cols))
        v_loss = validate(model, val_loader, criterion, device, config['data']['output_window'], len(feature_cols))
        logger.info(f"Epoch {epoch+1}, Train Loss: {t_loss:.4f}, Val Loss: {v_loss:.4f}")

    results = evaluate(model, test_loader, device, config['data']['output_window'], feature_cols, target_cols)
    for name, mae, rmse, r2 in results:
        logger.info(f"Results for {name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

if __name__ == "__main__":
    main()