# fl_client.py
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
import numpy as np
import logging as log
import sys
from typing import Dict, List, Tuple

log.basicConfig(level=log.DEBUG)

from train import load_config, load_data_client, build_model, train, evaluate

class FLClient(fl.client.NumPyClient):
    def __init__(self, config, place_id):
        self.config = config
        self.device = torch.device(config["training"]["device"])
        self.feature_cols = config["data"]["feature_cols"]
        self.target_cols = config["data"]["target_cols"]
        self.output_window = config["data"]["output_window"]
        self.model = build_model(config, self.device)
        self.train_loader, self.test_loader = load_data_client(config, place_id)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["training"]["learning_rate"])

    def get_parameters(self) -> List[np.ndarray]:
        """Restituisce i parametri del modello come lista di array NumPy."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Imposta i parametri del modello da una lista di array NumPy."""
        state_dict = dict(zip(
            self.model.state_dict().keys(),
            [torch.tensor(p, device=self.device) for p in parameters]
        ))
        self.model.load_state_dict(state_dict)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Training del modello locale."""
        log.debug("Training...")
        self.set_parameters(parameters)
        train(
            self.model, 
            self.train_loader, 
            self.criterion, 
            self.optimizer, 
            self.device,
            self.output_window, 
            len(self.feature_cols)
        )
        log.debug("Trained")
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Valutazione del modello locale."""
        self.set_parameters(parameters)
        results = evaluate(
            self.model,
            self.test_loader,
            self.device,
            self.output_window,
            self.feature_cols,
            self.target_cols
        )
        avg_mae = sum(r[1] for r in results) / len(results)
        avg_mse = sum(r[2] for r in results) / len(results)

        print("\n[Client Evaluation]")
        print("="*30)
        for name, mae, rmse, _ in results:
            print(f"Target: {name:<10} | MEAN SQUARE ERROR: {rmse:.4f} | MEAN AVERAGE ERROR: {mae:.4f}")
    
        return float(avg_mse), len(self.test_loader.dataset), {
            "mean_average_error": float(avg_mae), 
            "mean_square_error": float(avg_mse)
        }

if __name__ == "__main__":
    config = load_config("config.yaml")

    if len(sys.argv) < 3:
        print("Usage: python fl_client.py <place_id> <server_address>")
        sys.exit(1)

    place_id = int(sys.argv[1])
    server_address = sys.argv[2]

    client = FLClient(config, place_id)

    # Avvio del client Flower
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    ) 