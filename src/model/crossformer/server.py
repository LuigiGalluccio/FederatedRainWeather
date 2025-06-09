from flwr.server.strategy import FedAvg
from flwr.common import FitRes, EvaluateRes
from typing import Dict, List, Tuple, Optional
import flwr as fl
import logging

# Configurazione logging per vedere i dettagli
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def weighted_average(metrics):
    """Aggregate metrics from multiple clients using weighted average."""
    if not metrics:
        return {}
    
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}
    
    aggregated_metrics = {}
    for key in metrics[0][1].keys():
        aggregated_metrics[key] = sum(
            num_examples * client_metrics[key] for num_examples, client_metrics in metrics
        ) / total_examples

    return aggregated_metrics

class CustomFedAvg(FedAvg):
    """Strategia FedAvg personalizzata per stampare le metriche."""
    
    def __init__(self):
        super().__init__()
        self.round_num = 0
    
    def aggregate_fit(self, rnd: int, results: List[Tuple[fl.client.Client, FitRes]], failures: List[BaseException]) -> Optional[fl.common.Weights]:
        """Aggregazione dopo il training."""
        self.round_num = rnd
        logger.info(f"\n{'='*50}")
        logger.info(f"ROUND {rnd} - TRAINING COMPLETED")
        logger.info(f"Successfully trained clients: {len(results)}")
        logger.info(f"Failed clients: {len(failures)}")
        
        if failures:
            logger.warning(f"Training failures: {failures}")
        
        # Chiama l'aggregazione standard
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        
        logger.info(f"Aggregated weights from {len(results)} clients")
        logger.info(f"{'='*50}\n")
        
        return aggregated_weights
    
    def aggregate_evaluate(self, rnd: int, results: List[Tuple[fl.client.Client, EvaluateRes]], failures: List[BaseException]) -> Optional[float]:
        """Aggregazione dopo la valutazione."""
        logger.info(f"\n{'='*50}")
        logger.info(f"ROUND {rnd} - EVALUATION RESULTS")
        logger.info(f"{'='*50}")
        
        if not results:
            logger.warning("No evaluation results to aggregate")
            return None
        
        # Estrai le metriche da tutti i client
        metrics_list = []
        total_examples = 0
        
        for i, (client, eval_res) in enumerate(results):
            num_examples = eval_res.num_examples
            loss = eval_res.loss
            metrics = eval_res.metrics
            
            total_examples += num_examples
            metrics_list.append((num_examples, metrics))
            
            logger.info(f"Client {i+1}:")
            logger.info(f"  - Examples: {num_examples}")
            logger.info(f"  - Loss: {loss:.4f}")
            if metrics:
                for key, value in metrics.items():
                    logger.info(f"  - {key}: {value:.4f}")
        
        # Calcola metriche aggregate
        if metrics_list:
            aggregated_metrics = weighted_average(metrics_list)
            logger.info(f"\n{'='*30}")
            logger.info(f"AGGREGATED METRICS (Round {rnd})")
            logger.info(f"{'='*30}")
            logger.info(f"Total examples: {total_examples}")
            
            for key, value in aggregated_metrics.items():
                logger.info(f"Aggregated {key}: {value:.4f}")
        
        # Calcola loss aggregata
        total_loss = sum(eval_res.loss * eval_res.num_examples for _, eval_res in results)
        aggregated_loss = total_loss / total_examples
        
        logger.info(f"Aggregated loss: {aggregated_loss:.4f}")
        logger.info(f"{'='*50}\n")
        
        return aggregated_loss

# Strategia personalizzata
strategy = CustomFedAvg()

def main():
    logger.info("Starting Flower server...")
    
    # Avvio del server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        force_final_distributed_eval=False,
    )

if __name__ == "__main__":
    main()