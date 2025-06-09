from torch.utils.data import Dataset
import pandas as pd
import torch

class Dataset(Dataset):
    def __init__(self, data, input_window, output_window, feature_cols, target_cols, start_idx=0, end_idx=None):
        # Gestione robusta dell'input: accetta path o DataFrame
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()

        df = df[feature_cols + target_cols]
        self.data = df.values
        
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.input_window = input_window
        self.output_window = output_window

        if end_idx is None:
            end_idx = len(self.data)
        self.data = self.data[start_idx:end_idx]

    def __len__(self):
        return max(0, len(self.data) - self.input_window - self.output_window)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_window, :-len(self.target_cols)]
        y = self.data[idx + self.input_window : idx + self.input_window + self.output_window, -len(self.target_cols):]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
