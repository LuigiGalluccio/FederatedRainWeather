import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import os
from scipy.stats import pearsonr
import sys
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import personalizzati
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
from logger import Logger
from dataset import WeatherDataset # Importati solo quelli necessari
from model import myTransformer

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def get_single_file_df(file_path):
    # Caricamento con separatore ';' basato sul tuo esempio
    df = pd.read_csv(file_path, sep=';')
    
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    print(f"File caricato: {file_path} | Righe: {len(df)}")
    return df

def load_data_client(config, place_id, df_full):
    input_window = config['data']['input_window']
    output_window = config['data']['output_window']
    feature_cols = config['data']['feature_cols']
    target_cols = config['data']['target_cols']
    batch_size = config['training']['batch_size']

    # Filtro e ordinamento
    df_station = df_full[df_full['station_id'] == place_id].sort_values('datetime').reset_index(drop=True)

    if df_station.empty:
        return None, None, None

    total_len = len(df_station)
    print(f"DEBUG: Stazione {place_id} ha {total_len} righe totali.")
    train_end = int(0.7 * total_len)
    val_end = int(0.85 * total_len)

    train_ds = WeatherDataset(df_station, input_window, output_window, feature_cols, target_cols, 0, train_end)
    val_ds = WeatherDataset(df_station, input_window, output_window, feature_cols, target_cols, train_end, val_end)
    test_ds = WeatherDataset(df_station, input_window, output_window, feature_cols, target_cols, val_end)

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

# def train(model, train_loader, criterion, optimizer, device, output_window, feature_dim):
#     model.train()
#     epoch_loss = 0.0
#     for x, y in train_loader:
#         x, y = x.to(device), y.to(device)
#         pred = model(x)
#         loss = criterion(pred, y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#     return epoch_loss / len(train_loader)

def train(model, train_loader, criterion, optimizer, device, output_window, feature_dim):
    model.train()
    epoch_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        # 1. Calcolo della predizione
        pred = model(x)
        
        # 2. Calcolo della loss "punto per punto" (senza riduzione)
        # Assicurati che nel Client il criterio sia: nn.HuberLoss(reduction='none')
        loss_elements = criterion(pred, y)
        
        # 3. Definiamo i pesi: diamo più importanza ai momenti in cui piove (y > 0)
        # Esempio: se piove il peso è 5, se è asciutto il peso è 1
        # weights = torch.where(y > 0.0, 20.0, 1.0).to(device)
        weights = torch.where(y > 0.0, 2.0, 1.0).to(device)

        
        # 4. Applichiamo i pesi e facciamo la media finale
        loss = (loss_elements * weights).mean()
        
        # 5. Backpropagation standard con Gradient Clipping
        optimizer.zero_grad()
        loss.backward()
        
        # Clipping per evitare che il Transformer "impazzisca" con i pesi alti
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(train_loader)


def validate(model, val_loader, criterion, device, output_window, feature_dim):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            
            # 1. Predizione
            pred = model(x)
            
            # 2. Loss punto per punto (assicurati che criterion abbia reduction='none')
            loss_elements = criterion(pred, y)
            
            # 3. Applichiamo gli STESSI pesi del training per coerenza
            # Questo ti permette di vedere se il modello migliora sui picchi
            weights = torch.where(y > 0.0, 20.0, 1.0).to(device)
            loss = (loss_elements * weights).mean()
            
            val_loss += loss.item()
            
    return val_loss / len(val_loader)

# def validate(model, val_loader, criterion, device, output_window, feature_dim):
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for x, y in val_loader:
#             x, y = x.to(device), y.to(device)
#             pred = model(x)
#             loss = criterion(pred, y)
#             val_loss += loss.item()
#     return val_loss / len(val_loader)

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
        
        # Metriche standard
        mae = mean_absolute_error(targets_flat, preds_flat)
        rmse = np.sqrt(mean_squared_error(targets_flat, preds_flat))
        r2 = r2_score(targets_flat, preds_flat)
        
        # --- NUOVA METRICA: CORRELAZIONE DI PEARSON ---
        # La correlazione ci dice se il modello "segue il trend" (sale quando piove)
        if np.std(preds_flat) < 1e-6 or np.std(targets_flat) < 1e-6:
            corr = 0.0 # Se uno dei due è costante, la correlazione è 0
        else:
            corr, _ = pearsonr(targets_flat, preds_flat)
            
        results.append((target_name, mae, rmse, r2, corr))
        
    return results

# def evaluate(model, test_loader, device, output_window, feature_cols, target_cols):
#     model.eval()
#     all_preds, all_targets = [], []
#     with torch.no_grad():
#         for x, y in test_loader:
#             x, y = x.to(device), y.to(device)
#             pred = model(x)
#             all_preds.append(pred.cpu().numpy())
#             all_targets.append(y.cpu().numpy())

#     all_preds = np.concatenate(all_preds, axis=0)
#     all_targets = np.concatenate(all_targets, axis=0)

#     results = []
#     for i, target_name in enumerate(target_cols):
#         preds_flat = all_preds[:, :, i].reshape(-1)
#         targets_flat = all_targets[:, :, i].reshape(-1)
#         mae = mean_absolute_error(targets_flat, preds_flat)
#         rmse = np.sqrt(mean_squared_error(targets_flat, preds_flat))
#         r2 = r2_score(targets_flat, preds_flat)
#         results.append((target_name, mae, rmse, r2))
#     return results

def main():     
    config = load_config("config.yaml")
    logger = Logger()

    # Percorso del file unico
    file_path = "../../../data/events_dataset_scaled.csv"
    df_final = get_single_file_df(file_path)
    
    device = torch.device(config['training']['device'])
    feature_cols = config['data']['feature_cols']
    target_cols = config['data']['target_cols']
    
    # Esempio su stazione 1
    # target_id = 1
    # train_loader, val_loader, test_loader = load_data_client(config, target_id, df_final)
    
    # if train_loader is None:
    #     logger.error(f"Dati non trovati per la stazione {target_id}")
    #     return

    train_loader, val_loader, test_loader = load_data_client(config, place_id=1, df_full=df_final)

    model = build_model(config, device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=1e-4)


    for epoch in range(config['training']['epochs']):
        train_loss = train(model, train_loader, criterion, optimizer, device, config['data']['output_window'], len(feature_cols))
        val_loss = validate(model, val_loader, criterion, device, config['data']['output_window'], len(feature_cols))
        logger.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    results = evaluate(model, test_loader, device, config['data']['output_window'], feature_cols, target_cols)
    # for name, mae, rmse, r2 in results:
    #     logger.info(f"Results for {name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    for name, mae, rmse, r2, corr in results:
        logger.info(f"Results for {name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, Corr={corr:.4f}")

if __name__ == "__main__":
    main()
