
import time
import numpy as np
from typing import Dict, List, Any, Optional, Callable
import logging

from .core.base import (
    AspectWeaver, Aspect, Pointcut, ExecutionPhase, 
    FederatedClient, FederatedServer, JoinPoint
)
from .aspects.fair import FAIRComplianceAspect
from .aspects.security import SecurityAspect
from .aspects.provenance import ProvenanceAspect
from .aspects.institutional import InstitutionalPolicyAspect, Policy
from .utils.helpers import (
    setup_logging, create_experiment_config, validate_model_parameters,
    compute_model_similarity, normalize_weights
)

logger = logging.getLogger(__name__)

class AspectFL:
    """
    Main AspectFL framework for aspect-oriented federated learning
    
    This class provides the high-level API for creating and managing
    federated learning systems with integrated trust, compliance, and security aspects.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize AspectFL framework
        
        Args:
            config: Configuration dictionary for the framework
        """
        self.config = config or create_experiment_config()
        
        # Setup logging
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        log_file = self.config.get('logging', {}).get('file')
        setup_logging(log_level, log_file)
        
        # Initialize core components
        self.weaver = AspectWeaver()
        self.server = FederatedServer(self.weaver)
        self.clients: Dict[str, FederatedClient] = {}
        
        # Initialize aspects
        self._initialize_aspects()
        
        # Experiment tracking
        self.experiment_results = {
            'rounds': [],
            'global_metrics': [],
            'client_metrics': {},
            'aspect_metrics': {
                'fair_compliance': [],
                'security_scores': [],
                'provenance_quality': [],
                'policy_compliance': []
            }
        }
        
        logger.info("AspectFL framework initialized")
    
    def _initialize_aspects(self):
        """Initialize and register all aspects"""
        aspect_config = self.config.get('aspects', {})
        
        # FAIR Compliance Aspect
        if aspect_config.get('fair_compliance', {}).get('enabled', True):
            fair_config = aspect_config['fair_compliance']
            fair_aspect = FAIRComplianceAspect(
                weights=fair_config.get('weights'),
                threshold=fair_config.get('threshold', 0.7)
            )
            
            fair_pointcut = Pointcut(
                phases=[ExecutionPhase.DATA_LOADING, ExecutionPhase.LOCAL_TRAINING, 
                       ExecutionPhase.MODEL_UPDATE_SUBMISSION]
            )
            
            self.weaver.register_aspect(
                Aspect("FAIR_Compliance", fair_pointcut, fair_aspect, priority=8)
            )
        
        # Security Aspect
        if aspect_config.get('security', {}).get('enabled', True):
            security_config = aspect_config['security']
            security_aspect = SecurityAspect(security_config)
            
            security_pointcut = Pointcut(
                phases=[ExecutionPhase.DATA_LOADING, ExecutionPhase.LOCAL_TRAINING,
                       ExecutionPhase.MODEL_UPDATE_SUBMISSION, ExecutionPhase.AGGREGATION]
            )
            
            self.weaver.register_aspect(
                Aspect("Security", security_pointcut, security_aspect, priority=9)
            )
        
        # Provenance Aspect
        if aspect_config.get('provenance', {}).get('enabled', True):
            provenance_config = aspect_config['provenance']
            provenance_aspect = ProvenanceAspect(provenance_config)
            
            provenance_pointcut = Pointcut()  # Apply to all phases
            
            self.weaver.register_aspect(
                Aspect("Provenance", provenance_pointcut, provenance_aspect, priority=10)
            )
        
        # Institutional Policy Aspect
        if aspect_config.get('institutional_policy', {}).get('enabled', True):
            policy_config = aspect_config['institutional_policy']
            policies = policy_config.get('policies', [])
            policy_aspect = InstitutionalPolicyAspect(policies)
            
            policy_pointcut = Pointcut()  # Apply to all phases
            
            self.weaver.register_aspect(
                Aspect("Institutional_Policy", policy_pointcut, policy_aspect, priority=7)
            )
    
    def add_client(self, client_id: str, data: Any = None, **kwargs) -> FederatedClient:
        """
        Add a client to the federated learning system
        
        Args:
            client_id: Unique identifier for the client
            data: Training data for the client
            **kwargs: Additional client configuration
            
        Returns:
            FederatedClient instance
        """
        client = FederatedClient(client_id, self.weaver)
        
        # Set client data and configuration
        if data is not None:
            client.state.local_data = data
        
        for key, value in kwargs.items():
            if hasattr(client.state, key):
                setattr(client.state, key, value)
        
        self.clients[client_id] = client
        self.server.register_client(client)
        
        # Initialize client metrics tracking
        self.experiment_results['client_metrics'][client_id] = {
            'accuracy': [],
            'loss': [],
            'fair_scores': [],
            'security_scores': [],
            'participation_rounds': []
        }
        
        logger.info(f"Added client: {client_id}")
        return client
    
    def add_policy(self, policy: Policy):
        """
        Add an institutional policy to the system
        
        Args:
            policy: Policy instance to add
        """
        # Find the institutional policy aspect and add the policy
        for aspect in self.weaver.aspects:
            if aspect.name == "Institutional_Policy":
                aspect.advice.add_policy(policy)
                break
        
        logger.info(f"Added policy: {policy.name}")
    
    def run_federated_learning(self, num_rounds: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the federated learning process
        
        Args:
            num_rounds: Number of federated learning rounds to run
            
        Returns:
            Dictionary containing experiment results
        """
        if num_rounds is None:
            num_rounds = self.config['federated_learning']['num_rounds']
        
        logger.info(f"Starting federated learning for {num_rounds} rounds")
        
        # Initialize global model
        self._initialize_global_model()
        
        for round_num in range(num_rounds):
            logger.info(f"Starting round {round_num + 1}/{num_rounds}")
            
            # Execute federated learning round
            round_results = self._execute_round(round_num)
            
            # Store round results
            self.experiment_results['rounds'].append(round_results)
            
            # Update aspect metrics
            self._update_aspect_metrics(round_results)
            
            # Log round summary
            self._log_round_summary(round_num, round_results)
        
        # Generate final results
        final_results = self._generate_final_results()
        
        logger.info("Federated learning completed")
        return final_results
    
    def _initialize_global_model(self):
        """Initialize the global model"""
        model_config = self.config.get('model', {})
        num_features = self.config.get('data', {}).get('num_features', 10)
        
        # Simple linear model initialization
        if model_config.get('type', 'linear') == 'linear':
            self.server.global_model = np.random.normal(0, 0.1, num_features)
        else:
            # For neural networks, initialize with more complex structure
            layers = model_config.get('hidden_layers', [64, 32])
            total_params = num_features * layers[0]
            for i in range(len(layers) - 1):
                total_params += layers[i] * layers[i + 1]
            total_params += layers[-1] * self.config.get('data', {}).get('num_classes', 2)
            
            self.server.global_model = np.random.normal(0, 0.1, total_params)
        
        logger.info(f"Initialized global model with {len(self.server.global_model)} parameters")
    
    def _execute_round(self, round_num: int) -> Dict[str, Any]:
        """Execute a single federated learning round"""
        round_start_time = time.time()
        
        # Server-side round initialization
        server_init_result = self.server.execute_phase(
            ExecutionPhase.ROUND_COMPLETION,
            current_round=round_num,
            global_model=self.server.global_model
        )
        
        # Client selection
        selected_clients = self._select_clients()
        
        # Client-side local training
        client_updates = []
        client_weights = []
        
        for client_id in selected_clients:
            client = self.clients[client_id]
            
            # Distribute global model to client
            client.state.model_parameters = self.server.global_model.copy()
            client.state.current_round = round_num
            
            # Local training
            update = self._client_local_training(client)
            
            if update is not None:
                client_updates.append(update)
                # Weight by data size (simplified)
                weight = len(client.state.local_data) if hasattr(client.state.local_data, '__len__') else 1.0
                client_weights.append(weight)
        
        # Server-side aggregation
        if client_updates:
            aggregated_model = self._server_aggregation(client_updates, client_weights, round_num)
            self.server.global_model = aggregated_model
        
        # Evaluation
        global_metrics = self._evaluate_global_model(round_num)
        
        # Collect round results
        round_results = {
            'round': round_num,
            'duration': time.time() - round_start_time,
            'participating_clients': selected_clients,
            'global_metrics': global_metrics,
            'num_client_updates': len(client_updates),
            'aspect_results': self._collect_aspect_results()
        }
        
        return round_results
    
    def _select_clients(self) -> List[str]:
        """Select clients for the current round"""
        client_fraction = self.config['federated_learning'].get('client_fraction', 1.0)
        num_selected = max(1, int(len(self.clients) * client_fraction))
        
        # Simple random selection
        import random
        selected = random.sample(list(self.clients.keys()), num_selected)
        
        return selected
    
    def _client_local_training(self, client: FederatedClient) -> Optional[np.ndarray]:
        """Perform local training on a client"""
        try:
            # Data loading phase
            data_result = client.execute_phase(
                ExecutionPhase.DATA_LOADING,
                data_batch=client.state.local_data,
                metadata={'client_id': client.client_id}
            )
            
            # Local training phase
            training_result = client.execute_phase(
                ExecutionPhase.LOCAL_TRAINING,
                input_model=client.state.model_parameters,
                training_data=client.state.local_data,
                epochs=self.config['federated_learning'].get('local_epochs', 5),
                learning_rate=self.config['federated_learning'].get('learning_rate', 0.01)
            )
            
            # Simulate model update (in practice, this would be actual training)
            model_update = self._simulate_model_update(client)
            
            # Model update submission phase
            submission_result = client.execute_phase(
                ExecutionPhase.MODEL_UPDATE_SUBMISSION,
                model_update=model_update,
                client_id=client.client_id
            )
            
            return submission_result.state.get('model_update', model_update)
            
        except Exception as e:
            logger.error(f"Error in client {client.client_id} local training: {str(e)}")
            return None
    
    def _simulate_model_update(self, client: FederatedClient) -> np.ndarray:
        """Simulate model update (placeholder for actual training)"""
        # Add small random noise to simulate training update
        noise_scale = 0.01
        noise = np.random.normal(0, noise_scale, client.state.model_parameters.shape)
        
        # Simulate different update patterns for different clients
        if 'malicious' in client.client_id:
            # Malicious client: add large noise
            noise *= 10
        elif 'low_quality' in client.client_id:
            # Low quality client: add more noise
            noise *= 3
        
        return client.state.model_parameters + noise
    
    def _server_aggregation(self, client_updates: List[np.ndarray], 
                          client_weights: List[float], round_num: int) -> np.ndarray:
        """Perform server-side aggregation"""
        # Server aggregation phase
        aggregation_result = self.server.execute_phase(
            ExecutionPhase.AGGREGATION,
            client_models=client_updates,
            client_weights=normalize_weights(client_weights),
            current_round=round_num,
            aggregation_method='fedavg'
        )
        
        # Get aggregated model (may be modified by aspects)
        aggregated_model = aggregation_result.state.get('aggregated_model')
        
        if aggregated_model is not None:
            return aggregated_model
        
        # Fallback to simple FedAvg
        normalized_weights = normalize_weights(client_weights)
        aggregated = np.zeros_like(client_updates[0])
        
        for update, weight in zip(client_updates, normalized_weights):
            aggregated += weight * update
        
        return aggregated
    
    def _evaluate_global_model(self, round_num: int) -> Dict[str, float]:
        """Evaluate the global model"""
        # Server evaluation phase
        eval_result = self.server.execute_phase(
            ExecutionPhase.EVALUATION,
            global_model=self.server.global_model,
            round=round_num
        )
        
        # Simulate evaluation metrics
        # In practice, this would involve actual model evaluation
        base_accuracy = 0.7 + 0.2 * (round_num / self.config['federated_learning']['num_rounds'])
        base_loss = 1.0 - 0.5 * (round_num / self.config['federated_learning']['num_rounds'])
        
        # Add some noise
        accuracy = base_accuracy + np.random.normal(0, 0.02)
        loss = base_loss + np.random.normal(0, 0.05)
        
        metrics = {
            'accuracy': max(0, min(1, accuracy)),
            'loss': max(0, loss),
            'f1_score': max(0, min(1, accuracy + np.random.normal(0, 0.01)))
        }
        
        return metrics
    
    def _collect_aspect_results(self) -> Dict[str, Any]:
        """Collect results from all aspects"""
        aspect_results = {}
        
        # Get the latest execution log from the weaver
        if self.weaver.execution_log:
            latest_log = self.weaver.execution_log[-1]
            aspect_results['executed_aspects'] = latest_log.get('applicable_aspects', [])
        
        return aspect_results
    
    def _update_aspect_metrics(self, round_results: Dict[str, Any]):
        """Update aspect-specific metrics"""
        # This would extract aspect-specific metrics from round results
        # For now, simulate some metrics
        
        # FAIR compliance score
        fair_score = 0.6 + 0.3 * np.random.random()
        self.experiment_results['aspect_metrics']['fair_compliance'].append(fair_score)
        
        # Security score
        security_score = 0.7 + 0.2 * np.random.random()
        self.experiment_results['aspect_metrics']['security_scores'].append(security_score)
        
        # Provenance quality
        provenance_quality = 0.8 + 0.15 * np.random.random()
        self.experiment_results['aspect_metrics']['provenance_quality'].append(provenance_quality)
        
        # Policy compliance
        policy_compliance = 0.75 + 0.2 * np.random.random()
        self.experiment_results['aspect_metrics']['policy_compliance'].append(policy_compliance)
    
    def _log_round_summary(self, round_num: int, round_results: Dict[str, Any]):
        """Log summary of round results"""
        metrics = round_results['global_metrics']
        logger.info(f"Round {round_num + 1} completed:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Loss: {metrics['loss']:.4f}")
        logger.info(f"  Participating clients: {len(round_results['participating_clients'])}")
        logger.info(f"  Duration: {round_results['duration']:.2f}s")
    
    def _generate_final_results(self) -> Dict[str, Any]:
        """Generate final experiment results"""
        # Compute final metrics
        final_accuracy = self.experiment_results['rounds'][-1]['global_metrics']['accuracy']
        final_loss = self.experiment_results['rounds'][-1]['global_metrics']['loss']
        
        # Compute average aspect scores
        avg_fair_score = np.mean(self.experiment_results['aspect_metrics']['fair_compliance'])
        avg_security_score = np.mean(self.experiment_results['aspect_metrics']['security_scores'])
        avg_provenance_quality = np.mean(self.experiment_results['aspect_metrics']['provenance_quality'])
        avg_policy_compliance = np.mean(self.experiment_results['aspect_metrics']['policy_compliance'])
        
        final_results = {
            'experiment_config': self.config,
            'final_metrics': {
                'accuracy': final_accuracy,
                'loss': final_loss,
                'fair_compliance': avg_fair_score,
                'security_score': avg_security_score,
                'provenance_quality': avg_provenance_quality,
                'policy_compliance': avg_policy_compliance
            },
            'round_history': self.experiment_results['rounds'],
            'aspect_metrics_history': self.experiment_results['aspect_metrics'],
            'total_rounds': len(self.experiment_results['rounds']),
            'total_clients': len(self.clients),
            'weaver_execution_log': self.weaver.execution_log
        }
        
        return final_results
    
    def get_aspect_summary(self) -> Dict[str, Any]:
        """Get summary of aspect execution and results"""
        summary = {
            'registered_aspects': [aspect.name for aspect in self.weaver.aspects],
            'total_executions': len(self.weaver.execution_log),
            'execution_by_phase': {},
            'execution_by_aspect': {}
        }
        
        # Analyze execution log
        for log_entry in self.weaver.execution_log:
            phase = log_entry.get('phase', 'unknown')
            aspects = log_entry.get('applicable_aspects', [])
            
            # Count by phase
            if phase not in summary['execution_by_phase']:
                summary['execution_by_phase'][phase] = 0
            summary['execution_by_phase'][phase] += 1
            
            # Count by aspect
            for aspect in aspects:
                if aspect not in summary['execution_by_aspect']:
                    summary['execution_by_aspect'][aspect] = 0
                summary['execution_by_aspect'][aspect] += 1
        
        return summary
    
    def export_results(self, filepath: str, format: str = 'json'):
        """Export experiment results to file"""
        import json
        
        results = self._generate_final_results()
        
        if format == 'json':
            # Convert numpy arrays to lists for JSON serialization
            from .utils.helpers import serialize_numpy
            serializable_results = serialize_numpy(results)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results exported to {filepath}")

# Convenience functions for common use cases

def create_aspectfl_system(config: Dict[str, Any] = None) -> AspectFL:
    """Create a new AspectFL system with default configuration"""
    return AspectFL(config)

def run_simple_experiment(num_clients: int = 5, num_rounds: int = 10, 
                         dataset: str = 'synthetic') -> Dict[str, Any]:
    """Run a simple AspectFL experiment with default settings"""
    config = create_experiment_config(
        num_clients=num_clients,
        num_rounds=num_rounds
    )
    
    # Create system
    system = AspectFL(config)
    
    # Add clients with synthetic data
    for i in range(num_clients):
        client_id = f"client_{i}"
        # Generate synthetic data
        data = np.random.randn(100, config['data']['num_features'])
        system.add_client(client_id, data)
    
    # Run federated learning
    results = system.run_federated_learning()
    
    return results

