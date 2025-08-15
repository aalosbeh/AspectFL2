"""
Utility functions for AspectFL
Common utilities and helper functions
"""

import time
import json
import numpy as np
import hashlib
from typing import Dict, List, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

def compute_model_similarity(model1: np.ndarray, model2: np.ndarray) -> float:
    """Compute cosine similarity between two model parameter vectors"""
    if model1.shape != model2.shape:
        return 0.0
    
    # Flatten arrays
    flat1 = model1.flatten()
    flat2 = model2.flatten()
    
    # Compute cosine similarity
    dot_product = np.dot(flat1, flat2)
    norm1 = np.linalg.norm(flat1)
    norm2 = np.linalg.norm(flat2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def compute_data_quality_score(data: Any, metadata: Dict[str, Any] = None) -> float:
    """Compute data quality score based on various metrics"""
    if data is None:
        return 0.0
    
    quality_factors = []
    
    # Completeness
    if hasattr(data, '__len__'):
        completeness = 1.0 if len(data) > 0 else 0.0
        quality_factors.append(completeness)
    
    # Consistency (check for missing values if applicable)
    if hasattr(data, 'isnull'):
        missing_ratio = data.isnull().sum().sum() / data.size if data.size > 0 else 1.0
        consistency = 1.0 - missing_ratio
        quality_factors.append(consistency)
    
    # Timeliness (if timestamp available in metadata)
    if metadata and 'timestamp' in metadata:
        age_hours = (time.time() - metadata['timestamp']) / 3600
        timeliness = max(0, 1 - age_hours / (24 * 7))  # Decay over a week
        quality_factors.append(timeliness)
    
    # Validity (basic checks)
    validity = 1.0
    if isinstance(data, np.ndarray):
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            validity = 0.5
    quality_factors.append(validity)
    
    return np.mean(quality_factors) if quality_factors else 0.5

def normalize_weights(weights: List[float]) -> List[float]:
    """Normalize a list of weights to sum to 1.0"""
    total = sum(weights)
    if total == 0:
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]

def exponential_moving_average(values: List[float], alpha: float = 0.1) -> float:
    """Compute exponential moving average of values"""
    if not values:
        return 0.0
    
    ema = values[0]
    for value in values[1:]:
        ema = alpha * value + (1 - alpha) * ema
    
    return ema

def compute_entropy(probabilities: List[float]) -> float:
    """Compute Shannon entropy of probability distribution"""
    if not probabilities:
        return 0.0
    
    # Normalize probabilities
    total = sum(probabilities)
    if total == 0:
        return 0.0
    
    probs = [p / total for p in probabilities]
    
    # Compute entropy
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy

def detect_outliers(values: List[float], method: str = 'iqr', threshold: float = 1.5) -> List[int]:
    """Detect outliers in a list of values"""
    if len(values) < 4:
        return []
    
    values_array = np.array(values)
    outlier_indices = []
    
    if method == 'iqr':
        q1 = np.percentile(values_array, 25)
        q3 = np.percentile(values_array, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        outlier_indices = [i for i, v in enumerate(values) 
                          if v < lower_bound or v > upper_bound]
    
    elif method == 'zscore':
        mean_val = np.mean(values_array)
        std_val = np.std(values_array)
        
        if std_val > 0:
            z_scores = np.abs((values_array - mean_val) / std_val)
            outlier_indices = [i for i, z in enumerate(z_scores) if z > threshold]
    
    return outlier_indices

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    if denominator == 0:
        return default
    return numerator / denominator

def format_timestamp(timestamp: float) -> str:
    """Format timestamp as human-readable string"""
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))

def serialize_numpy(obj: Any) -> Any:
    """Serialize numpy arrays for JSON compatibility"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: serialize_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_numpy(item) for item in obj]
    else:
        return obj

def deserialize_numpy(obj: Any) -> Any:
    """Deserialize JSON-compatible objects back to numpy arrays where appropriate"""
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], (int, float)):
        return np.array(obj)
    elif isinstance(obj, dict):
        return {key: deserialize_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [deserialize_numpy(item) for item in obj]
    else:
        return obj

def compute_trust_decay(initial_trust: float, time_elapsed: float, 
                       half_life: float = 24 * 3600) -> float:
    """Compute trust decay over time using exponential decay"""
    decay_factor = np.exp(-time_elapsed * np.log(2) / half_life)
    return initial_trust * decay_factor

def aggregate_scores(scores: Dict[str, float], weights: Dict[str, float] = None) -> float:
    """Aggregate multiple scores with optional weights"""
    if not scores:
        return 0.0
    
    if weights is None:
        weights = {key: 1.0 for key in scores.keys()}
    
    weighted_sum = 0.0
    total_weight = 0.0
    
    for key, score in scores.items():
        weight = weights.get(key, 1.0)
        weighted_sum += score * weight
        total_weight += weight
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0

def validate_model_parameters(params: np.ndarray) -> Dict[str, Any]:
    """Validate model parameters and return validation results"""
    validation_results = {
        'is_valid': True,
        'issues': [],
        'statistics': {}
    }
    
    if params is None:
        validation_results['is_valid'] = False
        validation_results['issues'].append('Parameters are None')
        return validation_results
    
    if not isinstance(params, np.ndarray):
        validation_results['issues'].append('Parameters are not numpy array')
        try:
            params = np.array(params)
        except:
            validation_results['is_valid'] = False
            return validation_results
    
    # Check for NaN values
    if np.any(np.isnan(params)):
        validation_results['is_valid'] = False
        validation_results['issues'].append('Contains NaN values')
    
    # Check for infinite values
    if np.any(np.isinf(params)):
        validation_results['is_valid'] = False
        validation_results['issues'].append('Contains infinite values')
    
    # Compute statistics
    validation_results['statistics'] = {
        'shape': params.shape,
        'size': params.size,
        'mean': float(np.mean(params)),
        'std': float(np.std(params)),
        'min': float(np.min(params)),
        'max': float(np.max(params)),
        'norm': float(np.linalg.norm(params))
    }
    
    # Check for extreme values
    norm = validation_results['statistics']['norm']
    if norm > 1000:
        validation_results['issues'].append(f'Large parameter norm: {norm:.2f}')
    elif norm < 1e-6:
        validation_results['issues'].append(f'Very small parameter norm: {norm:.2e}')
    
    return validation_results

def create_secure_hash(data: Any, salt: str = "") -> str:
    """Create a secure hash of data with optional salt"""
    if isinstance(data, np.ndarray):
        data_bytes = data.tobytes()
    elif isinstance(data, str):
        data_bytes = data.encode('utf-8')
    elif isinstance(data, bytes):
        data_bytes = data
    else:
        data_bytes = str(data).encode('utf-8')
    
    # Add salt
    salted_data = salt.encode('utf-8') + data_bytes
    
    # Compute SHA-256 hash
    return hashlib.sha256(salted_data).hexdigest()

def rate_limit_check(last_request_time: float, min_interval: float) -> bool:
    """Check if enough time has passed since last request"""
    current_time = time.time()
    return (current_time - last_request_time) >= min_interval

def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """Calculate exponential backoff delay"""
    delay = base_delay * (2 ** attempt)
    return min(delay, max_delay)

class CircularBuffer:
    """Circular buffer for storing recent values"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.index = 0
        self.size = 0
    
    def append(self, value: Any):
        """Add a value to the buffer"""
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def get_values(self) -> List[Any]:
        """Get all values in chronological order"""
        if self.size < self.capacity:
            return [v for v in self.buffer[:self.size] if v is not None]
        else:
            return (self.buffer[self.index:] + self.buffer[:self.index])
    
    def get_recent(self, n: int) -> List[Any]:
        """Get the n most recent values"""
        values = self.get_values()
        return values[-n:] if len(values) >= n else values
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return self.size == self.capacity
    
    def clear(self):
        """Clear the buffer"""
        self.buffer = [None] * self.capacity
        self.index = 0
        self.size = 0

class MovingStatistics:
    """Compute moving statistics efficiently"""
    
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.values = CircularBuffer(window_size)
    
    def update(self, value: float):
        """Update with a new value"""
        self.values.append(value)
    
    def mean(self) -> float:
        """Get current moving average"""
        values = self.values.get_values()
        return np.mean(values) if values else 0.0
    
    def std(self) -> float:
        """Get current moving standard deviation"""
        values = self.values.get_values()
        return np.std(values) if len(values) > 1 else 0.0
    
    def min(self) -> float:
        """Get current moving minimum"""
        values = self.values.get_values()
        return np.min(values) if values else 0.0
    
    def max(self) -> float:
        """Get current moving maximum"""
        values = self.values.get_values()
        return np.max(values) if values else 0.0
    
    def percentile(self, p: float) -> float:
        """Get current moving percentile"""
        values = self.values.get_values()
        return np.percentile(values, p) if values else 0.0

def setup_logging(level: str = 'INFO', log_file: Optional[str] = None):
    """Setup logging configuration"""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def create_experiment_config(
    num_clients: int = 10,
    num_rounds: int = 10,
    local_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    **kwargs
) -> Dict[str, Any]:
    """Create a standard experiment configuration"""
    config = {
        'federated_learning': {
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'client_fraction': kwargs.get('client_fraction', 1.0),
            'local_epochs': local_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        },
        'aspects': {
            'fair_compliance': {
                'enabled': kwargs.get('fair_enabled', True),
                'threshold': kwargs.get('fair_threshold', 0.7),
                'weights': kwargs.get('fair_weights', {'F': 0.25, 'A': 0.25, 'I': 0.25, 'R': 0.25})
            },
            'security': {
                'enabled': kwargs.get('security_enabled', True),
                'anomaly_sensitivity': kwargs.get('anomaly_sensitivity', 2.0),
                'epsilon': kwargs.get('epsilon', 1.0),
                'delta': kwargs.get('delta', 1e-5)
            },
            'provenance': {
                'enabled': kwargs.get('provenance_enabled', True),
                'trust_threshold': kwargs.get('trust_threshold', 0.5)
            },
            'institutional_policy': {
                'enabled': kwargs.get('policy_enabled', True),
                'policies': kwargs.get('policies', [])
            }
        },
        'data': {
            'dataset': kwargs.get('dataset', 'synthetic'),
            'num_features': kwargs.get('num_features', 10),
            'num_classes': kwargs.get('num_classes', 2),
            'data_distribution': kwargs.get('data_distribution', 'iid')
        },
        'model': {
            'type': kwargs.get('model_type', 'linear'),
            'hidden_layers': kwargs.get('hidden_layers', [64, 32]),
            'activation': kwargs.get('activation', 'relu')
        },
        'evaluation': {
            'metrics': kwargs.get('metrics', ['accuracy', 'loss', 'f1_score']),
            'eval_frequency': kwargs.get('eval_frequency', 1)
        }
    }
    
    return config

