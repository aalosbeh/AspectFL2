"""
Comprehensive Experiments for AspectFL Research Paper
Healthcare and Financial Use Cases with Comparative Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import time
import json
import logging
from typing import Dict, List, Any, Tuple

# Import AspectFL framework
import sys
sys.path.append('/home/ubuntu/aspectfl_project')

from aspectfl import AspectFL, create_experiment_config
from aspectfl.aspects.institutional import (
    create_data_governance_policy, create_privacy_policy, create_security_policy
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthcareExperiment:
    """Healthcare federated learning experiment with AspectFL"""
    
    def __init__(self, num_hospitals: int = 5, num_rounds: int = 10):
        self.num_hospitals = num_hospitals
        self.num_rounds = num_rounds
        self.results = {}
        
    def generate_healthcare_data(self) -> Dict[str, Any]:
        """Generate synthetic healthcare data for federated learning"""
        logger.info("Generating synthetic healthcare data")
        
        # Simulate medical diagnosis data
        # Features: age, blood_pressure, cholesterol, glucose, etc.
        n_features = 20
        n_samples_per_hospital = 1000
        
        hospital_data = {}
        
        for hospital_id in range(self.num_hospitals):
            # Create heterogeneous data distributions across hospitals
            if hospital_id < 2:
                # Urban hospitals: more diverse patient population
                X, y = make_classification(
                    n_samples=n_samples_per_hospital,
                    n_features=n_features,
                    n_informative=15,
                    n_redundant=3,
                    n_classes=2,
                    class_sep=0.8,
                    random_state=42 + hospital_id
                )
            else:
                # Rural hospitals: more homogeneous population
                X, y = make_classification(
                    n_samples=n_samples_per_hospital,
                    n_features=n_features,
                    n_informative=12,
                    n_redundant=5,
                    n_classes=2,
                    class_sep=1.2,
                    random_state=42 + hospital_id
                )
            
            # Add hospital-specific bias
            X[:, 0] += hospital_id * 0.5  # Age bias
            X[:, 1] += np.random.normal(0, 0.3, n_samples_per_hospital)  # Measurement noise
            
            hospital_data[f"hospital_{hospital_id}"] = {
                'features': X,
                'labels': y,
                'metadata': {
                    'hospital_type': 'urban' if hospital_id < 2 else 'rural',
                    'patient_count': n_samples_per_hospital,
                    'data_quality': 0.9 - hospital_id * 0.05,  # Quality decreases with hospital_id
                    'timestamp': time.time() - hospital_id * 3600  # Staggered data collection
                }
            }
        
        return hospital_data
    
    def create_healthcare_policies(self) -> List[Any]:
        """Create healthcare-specific policies"""
        policies = []
        
        # HIPAA compliance policy
        hipaa_policy = create_privacy_policy(
            "healthcare_consortium",
            priority=9,
            min_epsilon=2.0  # Stricter privacy for healthcare
        )
        policies.append(hipaa_policy)
        
        # Data governance policy
        governance_policy = create_data_governance_policy(
            "healthcare_consortium",
            priority=8,
            data_types=['medical', 'sensitive'],
            max_data_age_hours=72  # Fresh data requirement
        )
        policies.append(governance_policy)
        
        # Security policy
        security_policy = create_security_policy(
            "healthcare_consortium",
            priority=7,
            min_security_score=0.85  # High security requirement
        )
        policies.append(security_policy)
        
        return policies
    
    def run_experiment(self, with_aspects: bool = True) -> Dict[str, Any]:
        """Run healthcare federated learning experiment"""
        logger.info(f"Running healthcare experiment (with_aspects={with_aspects})")
        
        # Generate data
        hospital_data = self.generate_healthcare_data()
        
        # Create experiment configuration
        config = create_experiment_config(
            num_clients=self.num_hospitals,
            num_rounds=self.num_rounds,
            local_epochs=3,
            learning_rate=0.01,
            fair_enabled=with_aspects,
            security_enabled=with_aspects,
            provenance_enabled=with_aspects,
            policy_enabled=with_aspects
        )
        
        # Create AspectFL system
        system = AspectFL(config)
        
        # Add policies if aspects are enabled
        if with_aspects:
            policies = self.create_healthcare_policies()
            for policy in policies:
                system.add_policy(policy)
        
        # Add hospital clients
        for hospital_id, data in hospital_data.items():
            system.add_client(
                hospital_id,
                data=data['features'],
                labels=data['labels'],
                metadata=data['metadata'],
                data_quality=data['metadata']['data_quality']
            )
        
        # Run federated learning
        start_time = time.time()
        results = system.run_federated_learning()
        execution_time = time.time() - start_time
        
        # Extract key metrics
        final_metrics = results['final_metrics']
        round_history = results['round_history']
        
        experiment_results = {
            'scenario': 'healthcare',
            'with_aspects': with_aspects,
            'execution_time': execution_time,
            'final_accuracy': final_metrics['accuracy'],
            'final_loss': final_metrics['loss'],
            'accuracy_history': [r['global_metrics']['accuracy'] for r in round_history],
            'loss_history': [r['global_metrics']['loss'] for r in round_history],
            'num_hospitals': self.num_hospitals,
            'num_rounds': self.num_rounds
        }
        
        if with_aspects:
            experiment_results.update({
                'fair_compliance': final_metrics['fair_compliance'],
                'security_score': final_metrics['security_score'],
                'provenance_quality': final_metrics['provenance_quality'],
                'policy_compliance': final_metrics['policy_compliance'],
                'fair_history': results['aspect_metrics_history']['fair_compliance'],
                'security_history': results['aspect_metrics_history']['security_scores']
            })
        
        return experiment_results

class FinancialExperiment:
    """Financial fraud detection experiment with AspectFL"""
    
    def __init__(self, num_banks: int = 8, num_rounds: int = 15):
        self.num_banks = num_banks
        self.num_rounds = num_rounds
        self.results = {}
    
    def generate_financial_data(self) -> Dict[str, Any]:
        """Generate synthetic financial fraud detection data"""
        logger.info("Generating synthetic financial fraud detection data")
        
        # Features: transaction_amount, account_age, num_transactions, etc.
        n_features = 25
        n_samples_per_bank = 2000
        
        bank_data = {}
        
        for bank_id in range(self.num_banks):
            # Create different fraud patterns across banks
            if bank_id < 3:
                # Large banks: more sophisticated fraud patterns
                X, y = make_classification(
                    n_samples=n_samples_per_bank,
                    n_features=n_features,
                    n_informative=20,
                    n_redundant=3,
                    n_classes=2,
                    class_sep=0.6,
                    weights=[0.95, 0.05],  # 5% fraud rate
                    random_state=42 + bank_id
                )
            else:
                # Smaller banks: simpler fraud patterns
                X, y = make_classification(
                    n_samples=n_samples_per_bank,
                    n_features=n_features,
                    n_informative=15,
                    n_redundant=5,
                    n_classes=2,
                    class_sep=1.0,
                    weights=[0.97, 0.03],  # 3% fraud rate
                    random_state=42 + bank_id
                )
            
            # Add bank-specific characteristics
            X[:, 0] *= (1 + bank_id * 0.2)  # Transaction amount scaling
            X[:, 1] += bank_id * 0.3  # Regional bias
            
            # Simulate data quality issues for some banks
            if bank_id >= 6:  # Last 2 banks have quality issues
                noise_mask = np.random.random(X.shape) < 0.05
                X[noise_mask] += np.random.normal(0, 2, np.sum(noise_mask))
            
            bank_data[f"bank_{bank_id}"] = {
                'features': X,
                'labels': y,
                'metadata': {
                    'bank_size': 'large' if bank_id < 3 else 'small',
                    'transaction_count': n_samples_per_bank,
                    'fraud_rate': np.mean(y),
                    'data_quality': 0.95 - (bank_id >= 6) * 0.15,
                    'timestamp': time.time() - bank_id * 1800
                }
            }
        
        return bank_data
    
    def create_financial_policies(self) -> List[Any]:
        """Create financial sector-specific policies"""
        policies = []
        
        # PCI DSS compliance policy
        pci_policy = create_security_policy(
            "financial_consortium",
            priority=10,
            min_security_score=0.9  # Very high security for financial data
        )
        policies.append(pci_policy)
        
        # Data retention policy
        retention_policy = create_data_governance_policy(
            "financial_consortium",
            priority=8,
            data_types=['financial', 'transaction'],
            max_data_age_hours=24  # Fresh transaction data
        )
        policies.append(retention_policy)
        
        # Privacy policy for customer data
        privacy_policy = create_privacy_policy(
            "financial_consortium",
            priority=9,
            min_epsilon=1.5
        )
        policies.append(privacy_policy)
        
        return policies
    
    def run_experiment(self, with_aspects: bool = True) -> Dict[str, Any]:
        """Run financial fraud detection experiment"""
        logger.info(f"Running financial experiment (with_aspects={with_aspects})")
        
        # Generate data
        bank_data = self.generate_financial_data()
        
        # Create experiment configuration
        config = create_experiment_config(
            num_clients=self.num_banks,
            num_rounds=self.num_rounds,
            local_epochs=4,
            learning_rate=0.005,
            fair_enabled=with_aspects,
            security_enabled=with_aspects,
            provenance_enabled=with_aspects,
            policy_enabled=with_aspects,
            anomaly_sensitivity=1.5  # More sensitive for financial fraud
        )
        
        # Create AspectFL system
        system = AspectFL(config)
        
        # Add policies if aspects are enabled
        if with_aspects:
            policies = self.create_financial_policies()
            for policy in policies:
                system.add_policy(policy)
        
        # Add bank clients
        for bank_id, data in bank_data.items():
            system.add_client(
                bank_id,
                data=data['features'],
                labels=data['labels'],
                metadata=data['metadata'],
                data_quality=data['metadata']['data_quality']
            )
        
        # Run federated learning
        start_time = time.time()
        results = system.run_federated_learning()
        execution_time = time.time() - start_time
        
        # Extract key metrics
        final_metrics = results['final_metrics']
        round_history = results['round_history']
        
        experiment_results = {
            'scenario': 'financial',
            'with_aspects': with_aspects,
            'execution_time': execution_time,
            'final_accuracy': final_metrics['accuracy'],
            'final_loss': final_metrics['loss'],
            'accuracy_history': [r['global_metrics']['accuracy'] for r in round_history],
            'loss_history': [r['global_metrics']['loss'] for r in round_history],
            'num_banks': self.num_banks,
            'num_rounds': self.num_rounds
        }
        
        if with_aspects:
            experiment_results.update({
                'fair_compliance': final_metrics['fair_compliance'],
                'security_score': final_metrics['security_score'],
                'provenance_quality': final_metrics['provenance_quality'],
                'policy_compliance': final_metrics['policy_compliance'],
                'fair_history': results['aspect_metrics_history']['fair_compliance'],
                'security_history': results['aspect_metrics_history']['security_scores']
            })
        
        return experiment_results

class ComparativeAnalysis:
    """Comparative analysis of AspectFL vs traditional federated learning"""
    
    def __init__(self):
        self.results = {}
    
    def run_comprehensive_experiments(self) -> Dict[str, Any]:
        """Run comprehensive comparative experiments"""
        logger.info("Starting comprehensive comparative experiments")
        
        all_results = {}
        
        # Healthcare experiments
        logger.info("Running healthcare experiments")
        healthcare_exp = HealthcareExperiment(num_hospitals=5, num_rounds=10)
        
        healthcare_with_aspects = healthcare_exp.run_experiment(with_aspects=True)
        healthcare_without_aspects = healthcare_exp.run_experiment(with_aspects=False)
        
        all_results['healthcare'] = {
            'with_aspects': healthcare_with_aspects,
            'without_aspects': healthcare_without_aspects
        }
        
        # Financial experiments
        logger.info("Running financial experiments")
        financial_exp = FinancialExperiment(num_banks=8, num_rounds=15)
        
        financial_with_aspects = financial_exp.run_experiment(with_aspects=True)
        financial_without_aspects = financial_exp.run_experiment(with_aspects=False)
        
        all_results['financial'] = {
            'with_aspects': financial_with_aspects,
            'without_aspects': financial_without_aspects
        }
        
        # Generate comparative metrics
        comparison_metrics = self._generate_comparison_metrics(all_results)
        all_results['comparison_metrics'] = comparison_metrics
        
        return all_results
    
    def _generate_comparison_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative metrics between with/without aspects"""
        metrics = {}
        
        for scenario in ['healthcare', 'financial']:
            with_aspects = results[scenario]['with_aspects']
            without_aspects = results[scenario]['without_aspects']
            
            metrics[scenario] = {
                'accuracy_improvement': with_aspects['final_accuracy'] - without_aspects['final_accuracy'],
                'convergence_speed': self._calculate_convergence_speed(
                    with_aspects['accuracy_history'], 
                    without_aspects['accuracy_history']
                ),
                'stability_improvement': self._calculate_stability_improvement(
                    with_aspects['accuracy_history'],
                    without_aspects['accuracy_history']
                ),
                'aspect_benefits': {
                    'fair_compliance': with_aspects.get('fair_compliance', 0),
                    'security_score': with_aspects.get('security_score', 0),
                    'provenance_quality': with_aspects.get('provenance_quality', 0),
                    'policy_compliance': with_aspects.get('policy_compliance', 0)
                }
            }
        
        return metrics
    
    def _calculate_convergence_speed(self, with_aspects: List[float], 
                                   without_aspects: List[float]) -> float:
        """Calculate convergence speed improvement"""
        # Find round where 90% of final accuracy is reached
        target_with = 0.9 * with_aspects[-1]
        target_without = 0.9 * without_aspects[-1]
        
        rounds_with = next((i for i, acc in enumerate(with_aspects) if acc >= target_with), len(with_aspects))
        rounds_without = next((i for i, acc in enumerate(without_aspects) if acc >= target_without), len(without_aspects))
        
        return (rounds_without - rounds_with) / len(with_aspects)
    
    def _calculate_stability_improvement(self, with_aspects: List[float],
                                       without_aspects: List[float]) -> float:
        """Calculate stability improvement (lower variance is better)"""
        var_with = np.var(with_aspects[-5:])  # Variance in last 5 rounds
        var_without = np.var(without_aspects[-5:])
        
        return (var_without - var_with) / var_without if var_without > 0 else 0

def run_all_experiments():
    """Run all experiments and save results"""
    logger.info("Starting comprehensive AspectFL experiments")
    
    # Create results directory
    import os
    os.makedirs('/home/ubuntu/aspectfl_project/results', exist_ok=True)
    
    # Run comparative analysis
    analysis = ComparativeAnalysis()
    results = analysis.run_comprehensive_experiments()
    
    # Save results
    with open('/home/ubuntu/aspectfl_project/results/experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Experiments completed. Results saved to experiment_results.json")
    
    return results

if __name__ == "__main__":
    # Run experiments
    results = run_all_experiments()
    
    # Print summary
    print("\n=== EXPERIMENT SUMMARY ===")
    for scenario in ['healthcare', 'financial']:
        print(f"\n{scenario.upper()} SCENARIO:")
        with_aspects = results[scenario]['with_aspects']
        without_aspects = results[scenario]['without_aspects']
        
        print(f"  With AspectFL:    Accuracy = {with_aspects['final_accuracy']:.4f}")
        print(f"  Without AspectFL: Accuracy = {without_aspects['final_accuracy']:.4f}")
        print(f"  Improvement:      {(with_aspects['final_accuracy'] - without_aspects['final_accuracy']):.4f}")
        
        if 'fair_compliance' in with_aspects:
            print(f"  FAIR Compliance:  {with_aspects['fair_compliance']:.4f}")
            print(f"  Security Score:   {with_aspects['security_score']:.4f}")
            print(f"  Policy Compliance: {with_aspects['policy_compliance']:.4f}")

