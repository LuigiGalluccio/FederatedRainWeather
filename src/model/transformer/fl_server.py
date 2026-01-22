from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig
import flwr as fl
import logging
from flwr.server.strategy.fedprox import FedProx
from train import load_config
import matplotlib.pyplot as plt
  # funzione che costruisce il modello

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Starting Flower server...")
config = load_config("config.yaml")
rounds = config['training']['rounds']
print(f"Configured for {rounds} rounds of federated learning.")

def weighted_average(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)
    aggregated_metrics = {}
    for key in metrics[0][1].keys():
        aggregated_metrics[key] = sum(
            num_examples * client_metrics[key] for num_examples, client_metrics in metrics
        ) / total_examples
    return aggregated_metrics

def plot_metrics(history):
    """Genera i grafici per MAE e R2 lungo i round."""
    rounds = range(1, len(history.metrics_distributed["mae"]) + 1)
    
    # Estrazione valori MAE e R2
    mae_values = [val for _, val in history.metrics_distributed["mae"]]
    r2_values = [val for _, val in history.metrics_distributed["r2"]]

    plt.figure(figsize=(12, 5))

    # Subplot per MAE
    plt.subplot(1, 2, 1)
    plt.plot(rounds, mae_values, marker='o', color='b', label='MAE')
    plt.title('Andamento MAE (Global)')
    plt.xlabel('Round')
    plt.ylabel('Errore Medio Assoluto')
    plt.grid(True)
    plt.legend()

    # Subplot per R2
    plt.subplot(1, 2, 2)
    plt.plot(rounds, r2_values, marker='s', color='r', label='R2')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5) # Linea dello zero
    plt.title('Andamento R2 Score (Global)')
    plt.xlabel('Round')
    plt.ylabel('R2 Score')
    plt.grid(True)
    plt.legend()
    plt.savefig("federated_metrics.png")
    plt.tight_layout()
    plt.show()


history = fl.server.start_server(
    server_address="0.0.0.0:8081",
    config=fl.server.ServerConfig(num_rounds=rounds),
    strategy=FedProx(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        proximal_mu=1.0
    ),
)

if history:
    plot_metrics(history)