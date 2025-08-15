"""
AspectFL: Aspect-Oriented Programming for Federated Learning
Core base classes and infrastructure
"""

import uuid
import time
import hashlib
import numpy as np
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExecutionPhase(Enum):
    """Enumeration of federated learning execution phases"""
    DATA_LOADING = "data_loading"
    LOCAL_TRAINING = "local_training"
    MODEL_UPDATE_SUBMISSION = "model_update_submission"
    AGGREGATION = "aggregation"
    MODEL_DISTRIBUTION = "model_distribution"
    EVALUATION = "evaluation"
    ROUND_COMPLETION = "round_completion"

@dataclass
class JoinPoint:
    """Represents a join point in the federated learning execution"""
    timestamp: float
    context: str  # client_id or "server"
    phase: ExecutionPhase
    state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.id = str(uuid.uuid4())

class Pointcut:
    """Defines which join points an aspect should intercept"""
    
    def __init__(self, phases: List[ExecutionPhase] = None, 
                 contexts: List[str] = None,
                 condition: Callable[[JoinPoint], bool] = None):
        self.phases = phases or []
        self.contexts = contexts or []
        self.condition = condition or (lambda jp: True)
    
    def matches(self, join_point: JoinPoint) -> bool:
        """Check if this pointcut matches the given join point"""
        phase_match = not self.phases or join_point.phase in self.phases
        context_match = not self.contexts or join_point.context in self.contexts
        condition_match = self.condition(join_point)
        
        return phase_match and context_match and condition_match

class Advice(ABC):
    """Abstract base class for aspect advice"""
    
    @abstractmethod
    def execute(self, join_point: JoinPoint) -> JoinPoint:
        """Execute the advice at the given join point"""
        pass

class Aspect:
    """Represents an aspect with pointcut and advice"""
    
    def __init__(self, name: str, pointcut: Pointcut, advice: Advice, priority: int = 0):
        self.name = name
        self.pointcut = pointcut
        self.advice = advice
        self.priority = priority
        self.id = str(uuid.uuid4())
    
    def applies_to(self, join_point: JoinPoint) -> bool:
        """Check if this aspect applies to the given join point"""
        return self.pointcut.matches(join_point)

class AspectWeaver:
    """Weaves aspects into the federated learning execution flow"""
    
    def __init__(self):
        self.aspects: List[Aspect] = []
        self.execution_log: List[Dict[str, Any]] = []
    
    def register_aspect(self, aspect: Aspect):
        """Register an aspect with the weaver"""
        self.aspects.append(aspect)
        self.aspects.sort(key=lambda a: a.priority, reverse=True)
        logger.info(f"Registered aspect: {aspect.name} with priority {aspect.priority}")
    
    def unregister_aspect(self, aspect_name: str):
        """Unregister an aspect by name"""
        self.aspects = [a for a in self.aspects if a.name != aspect_name]
        logger.info(f"Unregistered aspect: {aspect_name}")
    
    def weave(self, join_point: JoinPoint) -> JoinPoint:
        """Weave applicable aspects at the given join point"""
        applicable_aspects = [a for a in self.aspects if a.applies_to(join_point)]
        
        # Log the weaving process
        self.execution_log.append({
            'timestamp': join_point.timestamp,
            'join_point_id': join_point.id,
            'phase': join_point.phase.value,
            'context': join_point.context,
            'applicable_aspects': [a.name for a in applicable_aspects]
        })
        
        # Execute aspects in priority order
        current_join_point = join_point
        for aspect in applicable_aspects:
            try:
                logger.debug(f"Executing aspect {aspect.name} at {join_point.phase.value}")
                current_join_point = aspect.advice.execute(current_join_point)
            except Exception as e:
                logger.error(f"Error executing aspect {aspect.name}: {str(e)}")
                # Continue with other aspects
        
        return current_join_point

@dataclass
class ProvenanceNode:
    """Represents a node in the provenance graph"""
    id: str
    type: str  # 'Data', 'Model', 'Activity', 'Agent'
    timestamp: float
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

@dataclass
class ProvenanceEdge:
    """Represents an edge in the provenance graph"""
    source_id: str
    target_id: str
    relation: str  # 'used', 'wasGeneratedBy', 'wasAssociatedWith', etc.
    timestamp: float
    attributes: Dict[str, Any] = field(default_factory=dict)

class ProvenanceGraph:
    """Maintains the provenance graph for federated learning"""
    
    def __init__(self):
        self.nodes: Dict[str, ProvenanceNode] = {}
        self.edges: List[ProvenanceEdge] = []
    
    def add_node(self, node: ProvenanceNode):
        """Add a node to the provenance graph"""
        self.nodes[node.id] = node
    
    def add_edge(self, source_id: str, target_id: str, relation: str, 
                 timestamp: float = None, attributes: Dict[str, Any] = None):
        """Add an edge to the provenance graph"""
        if timestamp is None:
            timestamp = time.time()
        
        edge = ProvenanceEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            timestamp=timestamp,
            attributes=attributes or {}
        )
        self.edges.append(edge)
    
    def get_provenance_chain(self, node_id: str) -> List[ProvenanceNode]:
        """Get the complete provenance chain for a node"""
        chain = []
        visited = set()
        
        def dfs(current_id):
            if current_id in visited or current_id not in self.nodes:
                return
            
            visited.add(current_id)
            chain.append(self.nodes[current_id])
            
            # Find all nodes that this node used
            for edge in self.edges:
                if edge.target_id == current_id and edge.relation == 'used':
                    dfs(edge.source_id)
        
        dfs(node_id)
        return chain
    
    def compute_trust_score(self, node_id: str) -> float:
        """Compute trust score based on provenance"""
        if node_id not in self.nodes:
            return 0.0
        
        chain = self.get_provenance_chain(node_id)
        if not chain:
            return 0.5  # Default trust score
        
        # Simple trust computation based on chain length and node types
        trust_score = 1.0
        for node in chain:
            if node.type == 'Data':
                # Data nodes contribute based on their quality attributes
                quality = node.attributes.get('quality', 0.8)
                trust_score *= quality
            elif node.type == 'Activity':
                # Activity nodes contribute based on their success rate
                success_rate = node.attributes.get('success_rate', 0.9)
                trust_score *= success_rate
        
        return min(trust_score, 1.0)

class FederatedLearningState:
    """Maintains the state of federated learning execution"""
    
    def __init__(self, client_id: str = None):
        self.client_id = client_id or "server"
        self.current_round = 0
        self.model_parameters: Optional[np.ndarray] = None
        self.local_data: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}
        self.provenance_graph = ProvenanceGraph()
        self.fair_compliance_score = 0.0
        self.security_log: List[Dict[str, Any]] = []
        self.policy_violations: List[Dict[str, Any]] = []
        self.communication_log: List[Dict[str, Any]] = []
        
        # FAIR compliance tracking
        self.fair_history: List[Tuple[float, float]] = []  # (timestamp, score)
        
        # Security monitoring
        self.anomaly_scores: List[float] = []
        self.threat_level = "LOW"
        
        # Performance metrics
        self.training_metrics: Dict[str, List[float]] = {
            'accuracy': [],
            'loss': [],
            'training_time': []
        }

class FederatedClient:
    """Represents a federated learning client with aspect weaving"""
    
    def __init__(self, client_id: str, weaver: AspectWeaver = None):
        self.client_id = client_id
        self.weaver = weaver or AspectWeaver()
        self.state = FederatedLearningState(client_id)
        self.is_active = True
    
    def execute_phase(self, phase: ExecutionPhase, **kwargs) -> Any:
        """Execute a federated learning phase with aspect weaving"""
        join_point = JoinPoint(
            timestamp=time.time(),
            context=self.client_id,
            phase=phase,
            state=kwargs,
            metadata={'client_id': self.client_id, 'round': self.state.current_round}
        )
        
        # Weave aspects
        enhanced_join_point = self.weaver.weave(join_point)
        
        # Update state based on enhanced join point
        for key, value in enhanced_join_point.state.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
        
        return enhanced_join_point

class FederatedServer:
    """Represents the federated learning server with aspect weaving"""
    
    def __init__(self, weaver: AspectWeaver = None):
        self.weaver = weaver or AspectWeaver()
        self.state = FederatedLearningState("server")
        self.clients: Dict[str, FederatedClient] = {}
        self.global_model: Optional[np.ndarray] = None
    
    def register_client(self, client: FederatedClient):
        """Register a client with the server"""
        self.clients[client.client_id] = client
        logger.info(f"Registered client: {client.client_id}")
    
    def execute_phase(self, phase: ExecutionPhase, **kwargs) -> Any:
        """Execute a server-side federated learning phase with aspect weaving"""
        join_point = JoinPoint(
            timestamp=time.time(),
            context="server",
            phase=phase,
            state=kwargs,
            metadata={'round': self.state.current_round, 'num_clients': len(self.clients)}
        )
        
        # Weave aspects
        enhanced_join_point = self.weaver.weave(join_point)
        
        # Update state based on enhanced join point
        for key, value in enhanced_join_point.state.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
        
        return enhanced_join_point

def compute_checksum(data: Any) -> str:
    """Compute SHA-256 checksum of data"""
    if isinstance(data, np.ndarray):
        data_bytes = data.tobytes()
    else:
        data_bytes = str(data).encode('utf-8')
    
    return hashlib.sha256(data_bytes).hexdigest()

def generate_uuid() -> str:
    """Generate a unique identifier"""
    return str(uuid.uuid4())

