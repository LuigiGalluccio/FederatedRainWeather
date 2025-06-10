from flwr.server.strategy import FedAvg, FedProx
from flwr.server import ServerConfig
import flwr as fl
from train import load_config

config = load_config("config.yaml")
rounds = config["training"]["rounds"]

def weighted_average(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)
    aggregated_metrics = {}
    for key in metrics[0][1].keys():
        aggregated_metrics[key] = sum(
            num_examples * client_metrics[key] for num_examples, client_metrics in metrics
        ) / total_examples
    return aggregated_metrics

fl.server.start_server(
    server_address="0.0.0.0:8082",
    config=ServerConfig(num_rounds=rounds),
    strategy=FedProx(
        fraction_fit=0.6,
        fraction_evaluate=0.6,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        proximal_mu=0.6
    ),
)