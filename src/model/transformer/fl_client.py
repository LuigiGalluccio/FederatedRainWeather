import torch
import flwr as fl
import sys
import logging
# Importiamo le funzioni dal tuo nuovo train.py
from train import get_single_file_df, load_config, load_data_client, build_model, train, evaluate, validate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FLClient(fl.client.NumPyClient):
    def __init__(self, config, place_id):
        self.config = config
        self.device = torch.device(config["training"]["device"])
        self.place_id = place_id
        
        # 1. Carichiamo i dati normalizzati globalmente in memoria
        # Ogni client ricostruisce lo stesso DF per coerenza di scaling
        df_full = get_single_file_df("../../../data/events_dataset_scaled.csv")
        
        # 2. Inizializziamo i loader specifici per questo place_id
        self.train_loader, self.val_loader, self.test_loader = load_data_client(config, place_id, df_full)
        if self.train_loader is not None:
            n_train = len(self.train_loader.dataset)
            n_val = len(self.val_loader.dataset)
            n_test = len(self.test_loader.dataset)
            logger.info(f" [Client {place_id}] Dati caricati con successo:")
            logger.info(f"    - Campioni Training: {n_train}")
            logger.info(f"    - Campioni Validazione: {n_val}")
            logger.info(f"    - Campioni Test: {n_test}")
            logger.info(f"    - Batch Size: {config['training']['batch_size']}")
        else:
            logger.error(f" [Client {place_id}] ATTENZIONE: Nessun dato trovato per questa stazione!")
        
        self.model = build_model(config, self.device)
        self.criterion = torch.nn.HuberLoss(delta=1.0, reduction='none')
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
        
        # Prendi le epoche dal config o usa un default
        local_epochs = self.config['training']['epochs']
        train_loss = 0.0
        
        logger.info(f" [Client {self.place_id}] Inizio Round - Training locale...")

        for epoch in range(local_epochs):
            train_loss = train(self.model, self.train_loader, self.criterion, self.optimizer, 
                            self.device, self.output_window, self.input_dim)
            
            # AGGIUNTA: Validazione a ogni epoca per vedere se il modello locale "impazzisce"
            val_loss = validate(self.model, self.val_loader, self.criterion, self.device, 
                                self.output_window, self.input_dim)
            
            logger.info(f" [Client {self.place_id}] Epoch {epoch+1}/{local_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Ritorna i parametri e la loss dell'ultima epoca
        return self.get_parameters(), len(self.train_loader.dataset), {"loss": float(train_loss)}


    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        results = evaluate(self.model, self.test_loader, self.device, 
                        self.output_window, self.feature_cols, self.target_cols)
        
        # results è ora: [(target_name, mae, rmse, r2, corr), ...]
        avg_mae = sum(r[1] for r in results) / len(results)
        avg_r2 = sum(r[3] for r in results) / len(results)
        avg_corr = sum(r[4] for r in results) / len(results) # Nuovo!
        
        return float(avg_mae), len(self.test_loader.dataset), {
            "mae": float(avg_mae),
            "r2": float(avg_r2),
            "pearson": float(avg_corr) # Invialo al server
        }

if __name__ == "__main__":
    config = load_config("config.yaml")
    place_id = int(sys.argv[1])
    server_address = sys.argv[2] if len(sys.argv) > 2 else "0.0.0.0:8081"
    
    client = FLClient(config, place_id)
    fl.client.start_client(server_address=server_address, client=client.to_client())