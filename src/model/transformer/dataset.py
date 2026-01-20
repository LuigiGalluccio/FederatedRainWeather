# dataset.py
import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np

# class WeatherDataset(Dataset):
#     def __init__(self, df, input_window, output_window, feature_cols, target_col="RainRate"):
#         """
#         df: DataFrame già downsampled ogni 10 minuti
#         input_window: numero di step di input (es. 6 = 1 ora di storico)
#         output_window: numero di step di output (es. 6 = prossima ora da predire)
#         feature_cols: lista di colonne da usare come input
#         target_col: colonna target da predire
#         """
#         self.df = df.copy()
#         self.input_window = input_window
#         self.output_window = output_window
#         self.feature_cols = feature_cols
#         self.target_col = target_col

#         # Array numpy con tutte le feature + target
#         self.data = df[feature_cols + [target_col]].values

#         # Indici validi per le finestre
#         self.valid_indices = [
#             i for i in range(len(self.data) - input_window - output_window + 1)
#         ]

#     def __len__(self):
#         return len(self.valid_indices)

#     def __getitem__(self, idx):
#         i = self.valid_indices[idx]
#         # Input: [input_window, n_features]
#         x = self.data[i:i+self.input_window, :-1]
#         # Output: [output_window, 1]
#         y = self.data[i+self.input_window:i+self.input_window+self.output_window, -1:]
#         return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class WeatherDataset(Dataset):
    def __init__(self,df,input_window,output_window,feature_cols,target_cols,start_idx=0,end_idx=None):
        """
        df: DataFrame completo già ordinato temporalmente
        input_window: lunghezza input
        output_window: lunghezza output
        feature_cols: colonne di input
        target_cols: colonne target
        start_idx: indice temporale iniziale
        end_idx: indice temporale finale (esclusivo)
        """

        self.df = df.reset_index(drop=True)

        self.input_window = input_window
        self.output_window = output_window
        self.feature_cols = feature_cols
        self.target_cols = target_cols

        self.data = self.df[feature_cols + target_cols].values

        if end_idx is None:
            end_idx = len(self.data)

        self.start_idx = start_idx
        self.end_idx = end_idx

        # finestre valide interamente contenute nello split
        self.valid_indices = list(
            range(
                start_idx,
                end_idx - input_window - output_window + 1
            )
        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]

        x = self.data[i : i + self.input_window, :len(self.feature_cols)]
        y = self.data[
            i + self.input_window :
            i + self.input_window + self.output_window,
            len(self.feature_cols):
        ]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )



def load_weather_data(base_path="storage/vantage-pro/2025", downsample_factor=60, station_id=None):
    """
    Carica tutti i CSV divisi per mese, unisce in un unico DataFrame,
    filtra le righe con RainRate > 0 e fa downsampling ogni 10 minuti.
    """
    month_folders = sorted(glob.glob(os.path.join(base_path, "*")))
    all_dfs = []

    for month_folder in month_folders:
        csv_files = sorted(glob.glob(os.path.join(month_folder, "*.csv")))
        if len(csv_files) == 0:
            print(f"Warning: nessun CSV trovato in {month_folder}")
            continue
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            all_dfs.append(df)

    if len(all_dfs) == 0:
        raise ValueError("Nessun CSV trovato! Controlla il path.")

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['Datetime'] = pd.to_datetime(full_df['Datetime'])

    if station_id is not None:
        full_df["station_id"] = station_id
        
    full_df = full_df.sort_values('Datetime').reset_index(drop=True)

    # Filtra solo righe con pioggia
    # df_rain = full_df[full_df['RainRate'] > 0].reset_index(drop=True)
    full_df['RainRate'] = full_df['RainRate'].clip(lower=0.0)
    full_df['RainRate'] = full_df['RainRate'].apply(lambda x: 0 if x < 0.1 else x)
    full_df['RainRate'] = np.log1p(full_df['RainRate'])

    df_zero = full_df[full_df['RainRate'] == 0].sample(frac=0.33)
    df_rain = full_df[full_df['RainRate'] > 0]
    full_df = pd.concat([df_zero, df_rain]).sort_values('Datetime')



    # Downsampling ogni 'downsample_factor' righe (10 minuti)
    return df_rain.iloc[::downsample_factor].reset_index(drop=True)



def preprocess_weather_data(df,scaler=None):

    df = df.copy()

    wind_rad = df['WindDir'] * np.pi / 180.0
    df['WindDir_sin'] = np.sin(wind_rad)
    df['WindDir_cos'] = np.cos(wind_rad)
    df = df.drop(columns=['WindDir'])

    cols_to_normalize = [
        'Barometer', 'TempIn', 'HumIn', 'TempOut', 
        'WindSpeed', 'HumOut', 'WindDir_sin', 'WindDir_cos'
    ]

    if scaler is None:
        scaler = StandardScaler()
        df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    else:
        df[cols_to_normalize] = scaler.transform(df[cols_to_normalize]) 

    return df, scaler