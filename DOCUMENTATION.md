# AspectFL Documentation

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [API Reference](#api-reference)
5. [Examples](#examples)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Install from PyPI (when available)

```bash
pip install aspectfl
```

### Install from Source

```bash
git clone https://github.com/aspectfl/aspectfl.git
cd aspectfl
pip install -r requirements.txt
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/aspectfl/aspectfl.git
cd aspectfl
pip install -r requirements.txt
pip install -e ".[dev,experiments]"
```

## Quick Start

### Basic Federated Learning with AspectFL

```python
from aspectfl import AspectFLFramework
from aspectfl.aspects import FAIRAspect, SecurityAspect
from aspectfl.utils import create_synthetic_clients

# Create synthetic federated learning clients
clients = create_synthetic_clients(
    num_clients=5,
    samples_per_client=1000,
    num_features=20
)

# Initialize AspectFL framework
framework = AspectFLFramework()

# Register aspects
framework.register_aspect(FAIRAspect(priority=1))
framework.register_aspect(SecurityAspect(priority=2))

# Run federated learning
results = framework.run_federated_learning(
    clients=clients,
    rounds=10,
    local_epochs=5
)

print(f"Final accuracy: {results['accuracy']:.4f}")
print(f"FAIR compliance: {results['fair_score']:.4f}")
```

## Core Concepts

### Aspects

AspectFL implements four core aspects that address cross-cutting concerns in federated learning:

#### 1. FAIR Compliance Aspect

Ensures adherence to FAIR (Findability, Accessibility, Interoperability, Reusability) principles:

```python
from aspectfl.aspects import FAIRAspect

fair_aspect = FAIRAspect(
    priority=1,
    metadata_requirements={
        'data_source': True,
        'collection_date': True,
        'quality_score': True
    },
    accessibility_timeout=30,
    interoperability_standards=['json', 'csv']
)
```

#### 2. Security Aspect

Provides comprehensive threat detection and privacy protection:

```python
from aspectfl.aspects import SecurityAspect

security_aspect = SecurityAspect(
    priority=2,
    anomaly_threshold=2.0,
    differential_privacy=True,
    epsilon=2.0,
    delta=1e-5
)
```

#### 3. Provenance Aspect

Maintains comprehensive audit trails and lineage tracking:

```python
from aspectfl.aspects import ProvenanceAspect

provenance_aspect = ProvenanceAspect(
    priority=3,
    track_data_lineage=True,
    track_model_evolution=True,
    store_intermediate_results=True
)
```

#### 4. Institutional Policy Aspect

Enables dynamic policy enforcement:

```python
from aspectfl.aspects import PolicyAspect

policy_aspect = PolicyAspect(
    priority=4,
    policy_file='policies.yaml',
    conflict_resolution='priority_based'
)
```

### Join Points

AspectFL intercepts federated learning execution at six critical join points:

1. **Data Loading**: When clients load their local datasets
2. **Local Training**: During local model training
3. **Model Update Submission**: When clients submit model updates
4. **Global Aggregation**: During server-side model aggregation
5. **Model Distribution**: When the global model is distributed
6. **Evaluation**: During model evaluation phases

### Aspect Weaver

The aspect weaver coordinates aspect execution using a priority-based model:

```python
from aspectfl.core import AspectWeaver

weaver = AspectWeaver()
weaver.register_aspect(fair_aspect)
weaver.register_aspect(security_aspect)

# Aspects are executed in priority order at each join point
```

## API Reference

### AspectFLFramework

Main framework class for running federated learning with aspects.

```python
class AspectFLFramework:
    def __init__(self, config=None):
        """Initialize the AspectFL framework."""
        
    def register_aspect(self, aspect):
        """Register an aspect with the framework."""
        
    def run_federated_learning(self, clients, rounds, local_epochs=5):
        """Run federated learning with registered aspects."""
        
    def get_metrics(self):
        """Get comprehensive metrics from all aspects."""
```

### Base Aspect Class

All aspects inherit from the base Aspect class:

```python
class Aspect:
    def __init__(self, priority=0):
        """Initialize aspect with priority."""
        
    def before_join_point(self, context):
        """Execute before join point."""
        
    def after_join_point(self, context):
        """Execute after join point."""
        
    def get_pointcuts(self):
        """Return list of applicable join points."""
```

### FAIR Compliance Aspect

```python
class FAIRAspect(Aspect):
    def __init__(self, priority=1, **kwargs):
        """Initialize FAIR compliance aspect."""
        
    def assess_findability(self, metadata):
        """Assess findability score."""
        
    def assess_accessibility(self, endpoints):
        """Assess accessibility score."""
        
    def assess_interoperability(self, formats):
        """Assess interoperability score."""
        
    def assess_reusability(self, documentation):
        """Assess reusability score."""
        
    def get_fair_score(self):
        """Get overall FAIR compliance score."""
```

### Security Aspect

```python
class SecurityAspect(Aspect):
    def __init__(self, priority=2, **kwargs):
        """Initialize security aspect."""
        
    def detect_anomalies(self, model_updates):
        """Detect anomalous model updates."""
        
    def apply_differential_privacy(self, data):
        """Apply differential privacy mechanisms."""
        
    def verify_integrity(self, data):
        """Verify data integrity."""
        
    def get_security_score(self):
        """Get overall security score."""
```

## Examples

### Healthcare Federated Learning

```python
from aspectfl import AspectFLFramework
from aspectfl.aspects import FAIRAspect, SecurityAspect, PolicyAspect
from experiments.healthcare_experiment import create_healthcare_clients

# Create healthcare clients (hospitals)
hospitals = create_healthcare_clients(
    num_hospitals=5,
    patients_per_hospital=1000,
    features=['age', 'blood_pressure', 'cholesterol', 'diagnosis']
)

# Configure aspects for healthcare compliance
fair_aspect = FAIRAspect(
    priority=1,
    metadata_requirements={
        'hipaa_compliance': True,
        'patient_consent': True,
        'data_anonymization': True
    }
)

security_aspect = SecurityAspect(
    priority=2,
    differential_privacy=True,
    epsilon=2.0,  # HIPAA-compliant privacy level
    integrity_checking=True
)

policy_aspect = PolicyAspect(
    priority=3,
    policies={
        'data_retention': '72_hours',
        'minimum_security_score': 0.85,
        'required_encryption': True
    }
)

# Initialize framework
framework = AspectFLFramework()
framework.register_aspect(fair_aspect)
framework.register_aspect(security_aspect)
framework.register_aspect(policy_aspect)

# Run federated learning
results = framework.run_federated_learning(
    clients=hospitals,
    rounds=10,
    local_epochs=5
)

print(f"Healthcare FL Results:")
print(f"  Accuracy: {results['accuracy']:.4f}")
print(f"  FAIR Compliance: {results['fair_score']:.4f}")
print(f"  Security Score: {results['security_score']:.4f}")
print(f"  Policy Compliance: {results['policy_compliance']:.4f}")
```

### Financial Federated Learning

```python
from aspectfl import AspectFLFramework
from aspectfl.aspects import FAIRAspect, SecurityAspect, ProvenanceAspect
from experiments.financial_experiment import create_financial_clients

# Create financial clients (banks)
banks = create_financial_clients(
    num_banks=8,
    transactions_per_bank=2000,
    features=['amount', 'account_age', 'transaction_frequency', 'is_fraud']
)

# Configure aspects for financial compliance
fair_aspect = FAIRAspect(
    priority=1,
    metadata_requirements={
        'pci_dss_compliance': True,
        'data_classification': True,
        'audit_trail': True
    }
)

security_aspect = SecurityAspect(
    priority=2,
    differential_privacy=True,
    epsilon=1.5,  # Financial privacy requirements
    anomaly_detection=True,
    threat_assessment=True
)

provenance_aspect = ProvenanceAspect(
    priority=3,
    track_all_operations=True,
    regulatory_compliance=True,
    audit_logging=True
)

# Initialize framework
framework = AspectFLFramework()
framework.register_aspect(fair_aspect)
framework.register_aspect(security_aspect)
framework.register_aspect(provenance_aspect)

# Run federated learning
results = framework.run_federated_learning(
    clients=banks,
    rounds=15,
    local_epochs=3
)

print(f"Financial FL Results:")
print(f"  AUC Score: {results['auc']:.4f}")
print(f"  FAIR Compliance: {results['fair_score']:.4f}")
print(f"  Security Score: {results['security_score']:.4f}")
print(f"  Provenance Quality: {results['provenance_score']:.4f}")
```

### Custom Aspect Development

```python
from aspectfl.core import Aspect, JoinPoint

class CustomComplianceAspect(Aspect):
    def __init__(self, priority=5, regulation_type='GDPR'):
        super().__init__(priority)
        self.regulation_type = regulation_type
        self.compliance_score = 0.0
        
    def get_pointcuts(self):
        return [
            JoinPoint.DATA_LOADING,
            JoinPoint.MODEL_AGGREGATION,
            JoinPoint.EVALUATION
        ]
        
    def before_join_point(self, context):
        if context.join_point == JoinPoint.DATA_LOADING:
            self._validate_data_consent(context.data)
        elif context.join_point == JoinPoint.MODEL_AGGREGATION:
            self._check_aggregation_compliance(context.model_updates)
            
    def after_join_point(self, context):
        if context.join_point == JoinPoint.EVALUATION:
            self._update_compliance_score(context.results)
            
    def _validate_data_consent(self, data):
        # Custom consent validation logic
        pass
        
    def _check_aggregation_compliance(self, model_updates):
        # Custom aggregation compliance logic
        pass
        
    def _update_compliance_score(self, results):
        # Custom compliance scoring logic
        pass

# Use custom aspect
custom_aspect = CustomComplianceAspect(priority=5, regulation_type='GDPR')
framework.register_aspect(custom_aspect)
```

## Configuration

### Framework Configuration

```python
config = {
    'logging': {
        'level': 'INFO',
        'file': 'aspectfl.log'
    },
    'security': {
        'default_epsilon': 2.0,
        'default_delta': 1e-5,
        'anomaly_threshold': 2.0
    },
    'fair': {
        'metadata_weights': {
            'findability': 0.25,
            'accessibility': 0.25,
            'interoperability': 0.25,
            'reusability': 0.25
        }
    },
    'provenance': {
        'storage_backend': 'json',
        'compression': True
    }
}

framework = AspectFLFramework(config=config)
```

### Policy Configuration

Create a `policies.yaml` file:

```yaml
policies:
  data_governance:
    - name: "Data Retention Policy"
      condition: "data_age > 72_hours"
      action: "delete_data"
      priority: 1
      
    - name: "Minimum Security Score"
      condition: "security_score < 0.85"
      action: "reject_participation"
      priority: 2
      
  privacy:
    - name: "Differential Privacy"
      condition: "always"
      action: "apply_dp_noise"
      parameters:
        epsilon: 2.0
        delta: 1e-5
      priority: 3
      
  compliance:
    - name: "FAIR Compliance Check"
      condition: "fair_score < 0.7"
      action: "require_metadata_update"
      priority: 4
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

```python
# Error: ModuleNotFoundError: No module named 'aspectfl'
# Solution: Install AspectFL properly
pip install -e .
```

#### 2. Aspect Registration Issues

```python
# Error: Aspect not executing
# Solution: Check aspect priorities and pointcuts
aspect = FAIRAspect(priority=1)
print(aspect.get_pointcuts())  # Verify join points
```

#### 3. Performance Issues

```python
# Error: Slow execution with many aspects
# Solution: Optimize aspect execution order
# Higher priority aspects execute first
security_aspect = SecurityAspect(priority=1)  # Critical
fair_aspect = FAIRAspect(priority=2)          # Important
provenance_aspect = ProvenanceAspect(priority=3)  # Logging
```

#### 4. Configuration Issues

```python
# Error: Configuration not loading
# Solution: Verify configuration format
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
framework = AspectFLFramework(config=config)
```

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or configure AspectFL logging
config = {
    'logging': {
        'level': 'DEBUG',
        'file': 'debug.log'
    }
}
framework = AspectFLFramework(config=config)
```

### Performance Monitoring

```python
# Monitor aspect execution times
results = framework.run_federated_learning(clients, rounds=10)
metrics = framework.get_metrics()

print("Aspect Execution Times:")
for aspect_name, timing in metrics['aspect_timings'].items():
    print(f"  {aspect_name}: {timing:.4f}s")
```

### Getting Help

- **Documentation**: [https://aspectfl.readthedocs.io/](https://aspectfl.readthedocs.io/)
- **Issues**: [https://github.com/aspectfl/aspectfl/issues](https://github.com/aspectfl/aspectfl/issues)
- **Discussions**: [https://github.com/aspectfl/aspectfl/discussions](https://github.com/aspectfl/aspectfl/discussions)
- **Email**: aspectfl-support@example.com

For research-related questions, please contact the authors directly or reference the research paper.

