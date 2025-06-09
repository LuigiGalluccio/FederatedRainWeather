from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig
import flwr as fl

NUM_ROUNDS = 20

def weighted_average(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)
    aggregated_metrics = {}

    for key in metrics[0][1].keys():
        aggregated_metrics[key] = sum(
            num_examples * client_metrics[key] for num_examples, client_metrics in metrics
        ) / total_examples

    return aggregated_metrics

fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    ),
)