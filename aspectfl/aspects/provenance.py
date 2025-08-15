"""
Provenance Tracking Aspects for AspectFL
Implements comprehensive provenance tracking and provenance-aware aggregation
"""

import time
import uuid
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import networkx as nx
import logging

from ..core.base import Advice, JoinPoint, ExecutionPhase, ProvenanceNode, ProvenanceGraph

logger = logging.getLogger(__name__)

@dataclass
class ProvenanceRecord:
    """Detailed provenance record for federated learning operations"""
    id: str
    timestamp: float
    activity_type: str
    agent_id: str
    inputs: List[str]  # IDs of input entities
    outputs: List[str]  # IDs of output entities
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    quality_metrics: Dict[str, float]
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

class ProvenanceTrackingAdvice(Advice):
    """Advice for comprehensive provenance tracking"""
    
    def __init__(self):
        self.provenance_records: List[ProvenanceRecord] = []
        self.entity_registry: Dict[str, Dict[str, Any]] = {}
    
    def execute(self, join_point: JoinPoint) -> JoinPoint:
        """Track provenance at join point"""
        # Create provenance record for current activity
        record = self._create_provenance_record(join_point)
        self.provenance_records.append(record)
        
        # Update provenance graph
        self._update_provenance_graph(join_point, record)
        
        # Compute provenance scores
        provenance_scores = self._compute_provenance_scores(join_point)
        
        # Update join point state
        join_point.state['provenance_record'] = record
        join_point.state['provenance_scores'] = provenance_scores
        join_point.state['provenance_quality'] = self._assess_provenance_quality(record)
        
        logger.debug(f"Provenance tracked for {join_point.phase.value}: {record.id}")
        
        return join_point
    
    def _create_provenance_record(self, join_point: JoinPoint) -> ProvenanceRecord:
        """Create detailed provenance record"""
        # Identify inputs and outputs based on phase
        inputs, outputs = self._identify_inputs_outputs(join_point)
        
        # Extract parameters and metadata
        parameters = self._extract_parameters(join_point)
        metadata = self._extract_metadata(join_point)
        
        # Compute quality metrics
        quality_metrics = self._compute_quality_metrics(join_point)
        
        record = ProvenanceRecord(
            id=str(uuid.uuid4()),
            timestamp=join_point.timestamp,
            activity_type=join_point.phase.value,
            agent_id=join_point.context,
            inputs=inputs,
            outputs=outputs,
            parameters=parameters,
            metadata=metadata,
            quality_metrics=quality_metrics
        )
        
        return record
    
    def _identify_inputs_outputs(self, join_point: JoinPoint) -> Tuple[List[str], List[str]]:
        """Identify input and output entities for the activity"""
        inputs = []
        outputs = []
        
        if join_point.phase == ExecutionPhase.DATA_LOADING:
            # Inputs: data sources
            data_sources = join_point.state.get('data_sources', [])
            inputs.extend([self._register_entity('data_source', ds) for ds in data_sources])
            
            # Outputs: loaded dataset
            if 'loaded_dataset' in join_point.state:
                outputs.append(self._register_entity('dataset', join_point.state['loaded_dataset']))
        
        elif join_point.phase == ExecutionPhase.LOCAL_TRAINING:
            # Inputs: model, dataset, hyperparameters
            if 'input_model' in join_point.state:
                inputs.append(self._register_entity('model', join_point.state['input_model']))
            if 'training_data' in join_point.state:
                inputs.append(self._register_entity('dataset', join_point.state['training_data']))
            if 'hyperparameters' in join_point.state:
                inputs.append(self._register_entity('hyperparameters', join_point.state['hyperparameters']))
            
            # Outputs: updated model
            if 'updated_model' in join_point.state:
                outputs.append(self._register_entity('model', join_point.state['updated_model']))
        
        elif join_point.phase == ExecutionPhase.AGGREGATION:
            # Inputs: client models
            client_models = join_point.state.get('client_models', [])
            inputs.extend([self._register_entity('model', model) for model in client_models])
            
            # Outputs: global model
            if 'global_model' in join_point.state:
                outputs.append(self._register_entity('model', join_point.state['global_model']))
        
        return inputs, outputs
    
    def _register_entity(self, entity_type: str, entity_data: Any) -> str:
        """Register an entity and return its ID"""
        entity_id = str(uuid.uuid4())
        
        # Compute entity fingerprint for deduplication
        fingerprint = self._compute_entity_fingerprint(entity_data)
        
        # Check if entity already exists
        for existing_id, existing_entity in self.entity_registry.items():
            if existing_entity.get('fingerprint') == fingerprint:
                return existing_id
        
        # Register new entity
        self.entity_registry[entity_id] = {
            'type': entity_type,
            'fingerprint': fingerprint,
            'data': entity_data,
            'created_at': time.time()
        }
        
        return entity_id
    
    def _compute_entity_fingerprint(self, entity_data: Any) -> str:
        """Compute unique fingerprint for entity"""
        import hashlib
        
        if isinstance(entity_data, np.ndarray):
            data_str = str(entity_data.shape) + str(np.sum(entity_data))
        elif isinstance(entity_data, dict):
            data_str = json.dumps(entity_data, sort_keys=True)
        else:
            data_str = str(entity_data)
        
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _extract_parameters(self, join_point: JoinPoint) -> Dict[str, Any]:
        """Extract relevant parameters from join point"""
        parameters = {}
        
        # Common parameters
        if 'learning_rate' in join_point.state:
            parameters['learning_rate'] = join_point.state['learning_rate']
        if 'batch_size' in join_point.state:
            parameters['batch_size'] = join_point.state['batch_size']
        if 'epochs' in join_point.state:
            parameters['epochs'] = join_point.state['epochs']
        
        # Phase-specific parameters
        if join_point.phase == ExecutionPhase.AGGREGATION:
            if 'aggregation_method' in join_point.state:
                parameters['aggregation_method'] = join_point.state['aggregation_method']
            if 'client_weights' in join_point.state:
                parameters['client_weights'] = join_point.state['client_weights']
        
        return parameters
    
    def _extract_metadata(self, join_point: JoinPoint) -> Dict[str, Any]:
        """Extract metadata from join point"""
        metadata = {
            'phase': join_point.phase.value,
            'context': join_point.context,
            'timestamp': join_point.timestamp,
            'round': join_point.state.get('current_round', 0)
        }
        
        # Add any additional metadata from join point
        if 'metadata' in join_point.state:
            metadata.update(join_point.state['metadata'])
        
        return metadata
    
    def _compute_quality_metrics(self, join_point: JoinPoint) -> Dict[str, float]:
        """Compute quality metrics for the activity"""
        quality_metrics = {}
        
        # Data quality metrics
        if 'data_quality' in join_point.state:
            quality_metrics['data_quality'] = join_point.state['data_quality']
        
        # Model performance metrics
        if 'accuracy' in join_point.state:
            quality_metrics['accuracy'] = join_point.state['accuracy']
        if 'loss' in join_point.state:
            quality_metrics['loss'] = join_point.state['loss']
        
        # Completeness metrics
        required_fields = ['inputs', 'outputs', 'parameters']
        completeness = sum(1 for field in required_fields if field in join_point.state) / len(required_fields)
        quality_metrics['completeness'] = completeness
        
        # Timeliness metric (how recent is the data)
        current_time = time.time()
        data_age = current_time - join_point.timestamp
        timeliness = max(0, 1 - data_age / (24 * 3600))  # Decay over 24 hours
        quality_metrics['timeliness'] = timeliness
        
        return quality_metrics
    
    def _update_provenance_graph(self, join_point: JoinPoint, record: ProvenanceRecord):
        """Update the provenance graph with new record"""
        if not hasattr(join_point.state, 'provenance_graph'):
            join_point.state['provenance_graph'] = ProvenanceGraph()
        
        graph = join_point.state['provenance_graph']
        
        # Add activity node
        activity_node = ProvenanceNode(
            id=record.id,
            type='Activity',
            timestamp=record.timestamp,
            attributes={
                'activity_type': record.activity_type,
                'agent_id': record.agent_id,
                'parameters': record.parameters,
                'quality_metrics': record.quality_metrics
            }
        )
        graph.add_node(activity_node)
        
        # Add edges for inputs (used relationship)
        for input_id in record.inputs:
            graph.add_edge(input_id, record.id, 'used', record.timestamp)
        
        # Add edges for outputs (wasGeneratedBy relationship)
        for output_id in record.outputs:
            graph.add_edge(record.id, output_id, 'wasGeneratedBy', record.timestamp)
    
    def _compute_provenance_scores(self, join_point: JoinPoint) -> Dict[str, float]:
        """Compute provenance-based trust and quality scores"""
        scores = {}
        
        # Trust score based on agent reputation
        agent_trust = self._compute_agent_trust(join_point.context)
        scores['agent_trust'] = agent_trust
        
        # Data lineage score
        lineage_score = self._compute_lineage_score(join_point)
        scores['lineage_quality'] = lineage_score
        
        # Freshness score
        freshness_score = self._compute_freshness_score(join_point)
        scores['freshness'] = freshness_score
        
        # Overall provenance score
        weights = {'trust': 0.4, 'lineage': 0.3, 'freshness': 0.3}
        overall_score = (
            weights['trust'] * agent_trust +
            weights['lineage'] * lineage_score +
            weights['freshness'] * freshness_score
        )
        scores['overall'] = overall_score
        
        return scores
    
    def _compute_agent_trust(self, agent_id: str) -> float:
        """Compute trust score for an agent based on history"""
        # Simple trust computation based on past performance
        agent_records = [r for r in self.provenance_records if r.agent_id == agent_id]
        
        if not agent_records:
            return 0.5  # Default trust for new agents
        
        # Compute average quality metrics
        quality_scores = []
        for record in agent_records[-10:]:  # Consider last 10 records
            avg_quality = np.mean(list(record.quality_metrics.values())) if record.quality_metrics else 0.5
            quality_scores.append(avg_quality)
        
        return np.mean(quality_scores) if quality_scores else 0.5
    
    def _compute_lineage_score(self, join_point: JoinPoint) -> float:
        """Compute data lineage quality score"""
        # Check completeness of lineage information
        lineage_completeness = 0.0
        
        if 'data_sources' in join_point.state:
            lineage_completeness += 0.3
        if 'transformation_history' in join_point.state:
            lineage_completeness += 0.3
        if 'quality_checks' in join_point.state:
            lineage_completeness += 0.4
        
        return lineage_completeness
    
    def _compute_freshness_score(self, join_point: JoinPoint) -> float:
        """Compute data freshness score"""
        current_time = time.time()
        data_timestamp = join_point.state.get('data_timestamp', join_point.timestamp)
        
        # Exponential decay with 24-hour half-life
        age_hours = (current_time - data_timestamp) / 3600
        freshness = np.exp(-age_hours / 24)
        
        return freshness
    
    def _assess_provenance_quality(self, record: ProvenanceRecord) -> float:
        """Assess overall quality of provenance record"""
        quality_factors = []
        
        # Completeness of record
        required_fields = ['inputs', 'outputs', 'parameters', 'metadata']
        completeness = sum(1 for field in required_fields if getattr(record, field)) / len(required_fields)
        quality_factors.append(completeness)
        
        # Quality of metadata
        metadata_quality = len(record.metadata) / 10.0  # Normalize by expected number of fields
        quality_factors.append(min(metadata_quality, 1.0))
        
        # Presence of quality metrics
        metrics_quality = len(record.quality_metrics) / 5.0  # Normalize by expected number of metrics
        quality_factors.append(min(metrics_quality, 1.0))
        
        return np.mean(quality_factors)

class ProvenanceAwareAggregationAdvice(Advice):
    """Advice for provenance-aware federated aggregation"""
    
    def __init__(self, trust_threshold: float = 0.5):
        self.trust_threshold = trust_threshold
    
    def execute(self, join_point: JoinPoint) -> JoinPoint:
        """Perform provenance-aware aggregation"""
        if join_point.phase != ExecutionPhase.AGGREGATION:
            return join_point
        
        client_models = join_point.state.get('client_models', [])
        client_weights = join_point.state.get('client_weights', [])
        provenance_scores = join_point.state.get('provenance_scores', {})
        
        if not client_models:
            return join_point
        
        # Compute provenance-aware weights
        pa_weights = self._compute_provenance_aware_weights(
            client_models, client_weights, provenance_scores
        )
        
        # Perform weighted aggregation
        aggregated_model = self._weighted_aggregation(client_models, pa_weights)
        
        # Update join point state
        join_point.state['aggregated_model'] = aggregated_model
        join_point.state['provenance_aware_weights'] = pa_weights
        join_point.state['aggregation_method'] = 'provenance_aware_fedavg'
        
        logger.info(f"Provenance-aware aggregation completed with {len(client_models)} clients")
        
        return join_point
    
    def _compute_provenance_aware_weights(self, client_models: List[Any], 
                                        client_weights: List[float],
                                        provenance_scores: Dict[str, Any]) -> List[float]:
        """Compute weights that incorporate provenance information"""
        if not client_weights:
            client_weights = [1.0] * len(client_models)
        
        pa_weights = []
        
        for i, (model, weight) in enumerate(zip(client_models, client_weights)):
            # Get provenance score for this client
            client_id = f"client_{i}"  # Simplified client identification
            client_prov_scores = provenance_scores.get(client_id, {})
            
            # Extract relevant scores
            trust_score = client_prov_scores.get('agent_trust', 0.5)
            quality_score = client_prov_scores.get('lineage_quality', 0.5)
            freshness_score = client_prov_scores.get('freshness', 0.5)
            
            # Compute combined provenance score
            prov_score = 0.4 * trust_score + 0.3 * quality_score + 0.3 * freshness_score
            
            # Apply trust threshold
            if prov_score < self.trust_threshold:
                prov_score *= 0.1  # Heavily downweight untrusted clients
            
            # Combine with original weight
            pa_weight = weight * prov_score
            pa_weights.append(pa_weight)
        
        # Normalize weights
        total_weight = sum(pa_weights)
        if total_weight > 0:
            pa_weights = [w / total_weight for w in pa_weights]
        else:
            pa_weights = [1.0 / len(client_models)] * len(client_models)
        
        return pa_weights
    
    def _weighted_aggregation(self, client_models: List[np.ndarray], 
                            weights: List[float]) -> np.ndarray:
        """Perform weighted aggregation of client models"""
        if not client_models:
            return np.array([])
        
        # Convert models to numpy arrays if needed
        models = []
        for model in client_models:
            if isinstance(model, np.ndarray):
                models.append(model)
            else:
                models.append(np.array(model))
        
        # Weighted average
        aggregated = np.zeros_like(models[0])
        for model, weight in zip(models, weights):
            aggregated += weight * model
        
        return aggregated

class ProvenanceQueryAdvice(Advice):
    """Advice for querying and analyzing provenance information"""
    
    def execute(self, join_point: JoinPoint) -> JoinPoint:
        """Execute provenance queries and analysis"""
        query_type = join_point.state.get('provenance_query_type')
        
        if query_type == 'lineage':
            result = self._query_lineage(join_point)
        elif query_type == 'impact':
            result = self._query_impact(join_point)
        elif query_type == 'quality':
            result = self._query_quality(join_point)
        else:
            result = self._default_analysis(join_point)
        
        join_point.state['provenance_query_result'] = result
        
        return join_point
    
    def _query_lineage(self, join_point: JoinPoint) -> Dict[str, Any]:
        """Query the lineage of a specific entity"""
        entity_id = join_point.state.get('target_entity_id')
        if not entity_id:
            return {'error': 'No target entity specified'}
        
        graph = join_point.state.get('provenance_graph')
        if not graph:
            return {'error': 'No provenance graph available'}
        
        # Trace lineage backwards
        lineage = graph.get_provenance_chain(entity_id)
        
        return {
            'entity_id': entity_id,
            'lineage_length': len(lineage),
            'lineage_nodes': [{'id': node.id, 'type': node.type, 'timestamp': node.timestamp} 
                             for node in lineage]
        }
    
    def _query_impact(self, join_point: JoinPoint) -> Dict[str, Any]:
        """Query the impact of a specific entity or change"""
        entity_id = join_point.state.get('target_entity_id')
        if not entity_id:
            return {'error': 'No target entity specified'}
        
        # Find all entities that depend on this entity
        # This would require forward traversal of the provenance graph
        
        return {
            'entity_id': entity_id,
            'impact_analysis': 'Impact analysis not yet implemented'
        }
    
    def _query_quality(self, join_point: JoinPoint) -> Dict[str, Any]:
        """Query quality metrics across the provenance graph"""
        graph = join_point.state.get('provenance_graph')
        if not graph:
            return {'error': 'No provenance graph available'}
        
        # Aggregate quality metrics
        quality_summary = {
            'total_nodes': len(graph.nodes),
            'total_edges': len(graph.edges),
            'average_trust_score': 0.0,
            'quality_distribution': {}
        }
        
        return quality_summary
    
    def _default_analysis(self, join_point: JoinPoint) -> Dict[str, Any]:
        """Perform default provenance analysis"""
        graph = join_point.state.get('provenance_graph')
        if not graph:
            return {'error': 'No provenance graph available'}
        
        return {
            'graph_statistics': {
                'nodes': len(graph.nodes),
                'edges': len(graph.edges),
                'node_types': list(set(node.type for node in graph.nodes.values()))
            }
        }

class ProvenanceAspect:
    """Main provenance aspect that coordinates all provenance functionality"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize provenance advice components
        self.tracking_advice = ProvenanceTrackingAdvice()
        self.aggregation_advice = ProvenanceAwareAggregationAdvice(
            trust_threshold=self.config.get('trust_threshold', 0.5)
        )
        self.query_advice = ProvenanceQueryAdvice()
    
    def execute(self, join_point: JoinPoint) -> JoinPoint:
        """Execute comprehensive provenance management"""
        # Always track provenance
        join_point = self.tracking_advice.execute(join_point)
        
        # Apply provenance-aware aggregation if needed
        if join_point.phase == ExecutionPhase.AGGREGATION:
            join_point = self.aggregation_advice.execute(join_point)
        
        # Handle provenance queries if requested
        if 'provenance_query_type' in join_point.state:
            join_point = self.query_advice.execute(join_point)
        
        # Generate provenance summary
        self._generate_provenance_summary(join_point)
        
        return join_point
    
    def _generate_provenance_summary(self, join_point: JoinPoint):
        """Generate summary of provenance information"""
        summary = {
            'timestamp': join_point.timestamp,
            'phase': join_point.phase.value,
            'context': join_point.context,
            'provenance_tracked': 'provenance_record' in join_point.state,
            'quality_assessed': 'provenance_quality' in join_point.state,
            'scores_computed': 'provenance_scores' in join_point.state
        }
        
        join_point.state['provenance_summary'] = summary
        
        logger.debug(f"Provenance summary: {summary}")

