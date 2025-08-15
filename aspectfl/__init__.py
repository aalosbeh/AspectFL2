"""
AspectFL: Aspect-Oriented Programming for Federated Learning

A comprehensive framework that integrates trust, compliance, and security
aspects into federated learning systems using aspect-oriented programming principles.
"""

from .framework import AspectFL, create_aspectfl_system, run_simple_experiment
from .core.base import (
    AspectWeaver, Aspect, Pointcut, ExecutionPhase,
    FederatedClient, FederatedServer, JoinPoint
)
from .aspects.fair import FAIRComplianceAspect
from .aspects.security import SecurityAspect
from .aspects.provenance import ProvenanceAspect
from .aspects.institutional import InstitutionalPolicyAspect, Policy
from .utils.helpers import create_experiment_config, setup_logging

__version__ = "1.0.0"
__author__ = "AspectFL Research Team"
__email__ = "aspectfl@research.org"

__all__ = [
    'AspectFL',
    'create_aspectfl_system',
    'run_simple_experiment',
    'AspectWeaver',
    'Aspect',
    'Pointcut',
    'ExecutionPhase',
    'FederatedClient',
    'FederatedServer',
    'JoinPoint',
    'FAIRComplianceAspect',
    'SecurityAspect',
    'ProvenanceAspect',
    'InstitutionalPolicyAspect',
    'Policy',
    'create_experiment_config',
    'setup_logging'
]

