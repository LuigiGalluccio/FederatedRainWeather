import torch
import flwr as fl
import sys
import logging
# Importiamo le funzioni dal tuo nuovo train.py
from train import load_config, get_global_preprocessed_df, load_data_client, build_model, train, evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FLClient(fl.client.NumPyClient):
    def __init__(self, config, place_id):
        self.config = config
        self.device = torch.device(config["training"]["device"])
        
        # 1. Carichiamo i dati normalizzati globalmente in memoria
        # Ogni client ricostruisce lo stesso DF per coerenza di scaling
        df_full, _ = get_global_preprocessed_df(config)
        
        # 2. Inizializziamo i loader specifici per questo place_id
        self.train_loader, self.val_loader, self.test_loader = load_data_client(config, place_id, df_full)
        
        self.model = build_model(config, self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=config["training"]["learning_rate"], 
                                          weight_decay=1e-4)
        
        self.input_dim = len(config["data"]["feature_cols"])
        self.output_window = config["data"]["output_window"]
        self.feature_cols = config["data"]["feature_cols"]
        self.target_cols = config["data"]["target_cols"]

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), 
                              [torch.tensor(p, device=self.device) for p in parameters]))
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.train_loader, self.criterion, self.optimizer, 
              self.device, self.output_window, self.input_dim)
        
        # Monitoraggio metriche durante il training
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        results = evaluate(self.model, self.test_loader, self.device, 
                           self.output_window, self.feature_cols, self.target_cols)
        
        # Calcoliamo una media delle perdite (MSE è il terzo valore restituito da evaluate: index 2)
        avg_mse = sum(r[2]**2 for r in results) / len(results)
        avg_mae = sum(r[1] for r in results) / len(results)
        
        return float(avg_mse), len(self.test_loader.dataset), {"mae": float(avg_mae)}

if __name__ == "__main__":
    config = load_config("config.yaml")
    place_id = int(sys.argv[1])
    server_address = sys.argv[2] if len(sys.argv) > 2 else "0.0.0.0:8081"
    
    client = FLClient(config, place_id)
    fl.client.start_client(server_address=server_address, client=client.to_client())