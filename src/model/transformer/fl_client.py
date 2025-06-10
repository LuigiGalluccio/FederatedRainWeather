# fl_client.py
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import flwr as fl
import logging
import sys
from train import load_config, load_data_client, build_model, train, evaluate, validate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FLClient(fl.client.NumPyClient):
    def __init__(self, config,place_id):
        self.config = config
        self.device = torch.device(config["training"]["device"])
        self.feature_cols = config["data"]["feature_cols"]
        self.target_cols = config["data"]["target_cols"]
        self.output_window = config["data"]["output_window"]
        self.input_dim = config["model"]["input_dim"]
        self.output_dim = config["model"]["output_dim"]
        self.model = build_model(config, self.device)
        self.train_loader, self.val_loader, self.test_loader = load_data_client(config,place_id)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["training"]["learning_rate"],weight_decay=1e-4)

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(),[torch.tensor(p, device=self.device) for p in parameters]))
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, 
              self.train_loader, 
              self.criterion, 
              self.optimizer, 
              self.device, 
              self.output_window, 
              self.input_dim)
        val_results = evaluate(self.model,
                               self.test_loader,
                               self.device,
                               self.output_window,
                               self.feature_cols,
                               self.target_cols)
        avg_val_mae = sum(r[1] for r in val_results) / len(val_results)
        avg_val_mse = sum(r[2]**2 for r in val_results) / len(val_results)
        metrics = {"val_mae": avg_val_mae, "val_mse": avg_val_mse}
        torch.save(self.model.state_dict(), f"../../../data/trained_model/client_{place_id}_transformer.pth")
        return self.get_parameters(), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
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
        print("\n[Client Evaluation] TRANSFORMER")
        print("="*30)
        for name, mae , rmse, _ in results:
            print(f"Target: {name:<10} | MEAN SQUARE ERROR: {rmse:.4f} | MEAN AVERAGE ERROR: {mae:.4f}")
        return float(avg_mse), len(self.test_loader.dataset), {"mean average error": float(avg_mae), "mean square error": float(avg_mse)}

if __name__ == "__main__":
    config = load_config("config.yaml")
    logger.info("Starting Client...")
    if len(sys.argv) < 3:
        print("Usage: python fl_client.py <place_id> <server_address>")
        sys.exit(1)
    place_id = int(sys.argv[1])
    server_address = sys.argv[2]
    client = FLClient(config, place_id)
    fl.client.start_client(server_address=server_address, client=client.to_client())