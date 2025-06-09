# fl_client.py
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
import logging as log
import sys
log.basicConfig(level=log.INFO)

from train import load_config, load_data_client, build_model, train, evaluate

class FLClient(fl.client.NumPyClient):
    def __init__(self, config,place_id):
        self.config = config
        self.device = torch.device(config["training"]["device"])
        self.feature_cols = config["data"]["feature_cols"]
        self.target_cols = config["data"]["target_cols"]
        self.output_window = config["data"]["output_window"]
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
        train(self.model, self.train_loader, self.criterion, self.optimizer, self.device, self.output_window, len(self.feature_cols))
        val_results = evaluate(self.model, self.val_loader, self.device, self.output_window, self.feature_cols, self.target_cols)
        avg_val_mae = sum(r[1] for r in val_results) / len(val_results)
        print(f"[Client Validation] Avg MAE: {avg_val_mae:.4f}")
        return self.get_parameters(), len(self.train_loader.dataset), {}

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

    if len(sys.argv) < 2:
        print("Usage: python fl_client.py <place_id>")
        sys.exit(1)

    place_id = int(sys.argv[1]) 
    client = FLClient(config, place_id)
    fl.client.start_client(server_address="localhost:8080", client=client.to_client())