# AspectFL: Aspect-Oriented Programming for Trustworthy and Compliant Federated Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

AspectFL is a comprehensive aspect-oriented programming framework that addresses fundamental challenges in federated learning trustworthiness and compliance through systematic integration of cross-cutting concerns. Our framework implements four core aspects: FAIR compliance, security threat detection and mitigation, comprehensive provenance tracking, and institutional policy enforcement.

## Key Features

- **FAIR Compliance**: Automated monitoring and enforcement of Findability, Accessibility, Interoperability, and Reusability principles
- **Security**: Comprehensive threat detection, anomaly identification, and privacy-preserving mechanisms with differential privacy
- **Provenance Tracking**: Complete audit trails and lineage tracking for all federated learning activities
- **Policy Enforcement**: Dynamic enforcement of organization-specific policies and regulatory requirements
- **Aspect Weaving**: Sophisticated aspect weaver that intercepts federated learning execution at critical join points

## Research Paper

This repository accompanies our research paper "AspectFL: Aspect-Oriented Programming for Trustworthy and Compliant Federated Learning Systems" submitted to the Information journal (ISSN 2078-2489).

### Abstract

Federated learning has emerged as a paradigm-shifting approach for collaborative machine learning while preserving data privacy. However, existing federated learning frameworks face significant challenges in ensuring trustworthiness, regulatory compliance, and comprehensive security across heterogeneous institutional environments. We introduce AspectFL, a novel aspect-oriented programming framework that seamlessly integrates trust, compliance, and security concerns into federated learning systems through cross-cutting aspect weaving.

### Key Results

- **Healthcare**: 4.52% accuracy improvement with FAIR compliance score of 0.762
- **Financial**: 0.90% accuracy improvement with FAIR compliance score of 0.738
- **Security**: 94% detection rate for data poisoning, 89% for model poisoning
- **Policy Compliance**: 84.3% compliance rate across both domains

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

```
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
cryptography>=41.0.0
networkx>=3.1.0
```

## Quick Start

### Basic Usage

```python
from aspectfl import AspectFLFramework
from aspectfl.aspects import FAIRAspect, SecurityAspect, ProvenanceAspect, PolicyAspect

# Initialize AspectFL framework
framework = AspectFLFramework()

# Register aspects
framework.register_aspect(FAIRAspect(priority=1))
framework.register_aspect(SecurityAspect(priority=2))
framework.register_aspect(ProvenanceAspect(priority=3))
framework.register_aspect(PolicyAspect(priority=4))

# Run federated learning with aspects
results = framework.run_federated_learning(
    clients=clients,
    rounds=10,
    local_epochs=5
)
```

### Healthcare Example

```python
from experiments.healthcare_experiment import run_healthcare_experiment

# Run healthcare federated learning experiment
results = run_healthcare_experiment(
    num_hospitals=5,
    patients_per_hospital=1000,
    rounds=10,
    enable_aspects=True
)

print(f"Accuracy: {results['accuracy']:.4f}")
print(f"FAIR Compliance: {results['fair_score']:.4f}")
print(f"Security Score: {results['security_score']:.4f}")
```

### Financial Example

```python
from experiments.financial_experiment import run_financial_experiment

# Run financial federated learning experiment
results = run_financial_experiment(
    num_banks=8,
    transactions_per_bank=2000,
    rounds=15,
    enable_aspects=True
)

print(f"AUC Score: {results['auc']:.4f}")
print(f"FAIR Compliance: {results['fair_score']:.4f}")
print(f"Security Score: {results['security_score']:.4f}")
```

## Architecture

AspectFL employs a sophisticated aspect-oriented architecture with four core components:

### 1. FAIR Compliance Aspect
- Automated metadata generation and quality assessment
- Endpoint availability monitoring
- Standard format validation
- Documentation tracking

### 2. Security Aspect
- Statistical anomaly detection
- Differential privacy mechanisms
- Integrity verification
- Threat assessment and response

### 3. Provenance Aspect
- Comprehensive audit trails
- Lineage tracking
- W3C PROV-compliant provenance graph
- Query mechanisms for accountability

### 4. Institutional Policy Aspect
- Flexible policy definition language
- Conflict resolution mechanisms
- Real-time compliance monitoring
- Dynamic policy updates

## Repository Structure

```
aspectfl_project/
├── aspectfl/                 # Core framework
│   ├── core/                # Base classes and infrastructure
│   ├── aspects/             # Aspect implementations
│   ├── utils/               # Utility functions
│   └── framework.py         # Main framework
├── experiments/             # Experimental code
│   ├── comprehensive_experiments.py
│   ├── data_analysis.py
│   └── healthcare_experiment.py
├── data/                    
├── requirements.txt         # Python dependencies
├── setup.py                # Package setup
├── Makefile                # Build automation
├── LICENSE                 # MIT License
└── README.md               # This file
```

## Mathematical Framework

AspectFL provides formal mathematical foundations for AO federated learning:

### Aspect Weaving Model

For a federation F = {F₁, F₂, ..., Fₙ} with join points J = {j₁, j₂, ..., jₘ} and aspects A = {A₁, A₂, A₃, A₄}, the aspect weaving process transforms the execution context:

```
C'ₖ = αᵢₚ(αᵢₚ₋₁(...αᵢ₁(Cₖ)...))
```

### FAIR Compliance Scoring

```
S_FAIR = w_F · S_F + w_A · S_A + w_I · S_I + w_R · S_R
```

### Security Metrics

Anomaly detection score:
```
A(Δθᵢ) = ||Δθᵢ - μ_Δθ||₂ / σ_Δθ
```

Differential privacy:
```
θ̃ᵢ = θᵢ + N(0, σ²I)
where σ² = (2Δ² log(1.25/δ))/ε²
```

## Running Experiments

### Comprehensive Experiments

```bash
cd aspectfl_project
python experiments/comprehensive_experiments.py
```

### Data Analysis and Visualization

```bash
python experiments/data_analysis.py
```

### Individual Experiments

```bash
# Healthcare experiment
python -c "from experiments.healthcare_experiment import run_healthcare_experiment; run_healthcare_experiment()"

# Financial experiment  
python -c "from experiments.financial_experiment import run_financial_experiment; run_financial_experiment()"
```


### Development Setup

```bash
# Clone the repository
git clone https://github.com/alosbeh/aspectfl2.git
cd aspectfl

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black aspectfl/ experiments/
```

## Citation

If you use AspectFL in your research, please cite our paper:

```bibtex
@article{Alsobeh2025aspectfl,
  title={AspectFL: Aspect-Oriented Programming for Trustworthy and Compliant Federated Learning Systems},
  author={Anas AlSobeh, Amani Shatnawi, Aws Maqableh},
  journal={Information},
  volume={xx},
  number={xx},
  pages={xxx},
  year={2025},
  publisher={MDPI},
  doi={xxx}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


AspectFL: Enabling trustworthy and compliant federated learning through aspect-oriented programming.

