import os
import logging
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from deap import base, creator, tools
from typing import Dict, Tuple, Optional, Any
try:
    from torch.utils.tensorboard import SummaryWriter  # Optional
except Exception:
    SummaryWriter = None
    
# Import schema-driven evolution system
try:
    from training.schema_driven_evolution import SchemaEvolutionIntegrator
    SCHEMA_AVAILABLE = True
    logger_temp = logging.getLogger(__name__)
    logger_temp.info("Schema-driven evolution system available")
except ImportError as e:
    SCHEMA_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    # Use ASCII-safe warning to avoid encoding errors
    logger_temp.warning(f"Schema system not available: {e}")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# TensorBoard writer (optional; do not initialize at import time)
writer = None
if SummaryWriter is not None and os.environ.get("NEBULA_TENSORBOARD", "0") == "1":
    try:
        writer = SummaryWriter(log_dir='runs/nas_evo')
        logger.info("TensorBoard SummaryWriter initialized for NAS/EVO.")
    except Exception as e:
        logger.warning(f"Failed to initialize TensorBoard SummaryWriter: {e}")

# Define DEAP creator types
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# NAS: Neural Architecture Search with Schema Constraints
class NeuralArchitectureSearch(nn.Module):
    def __init__(self, input_dim, output_dim, device=None, schema_integrator=None):
        super(NeuralArchitectureSearch, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = []
        self.schema_integrator = schema_integrator
        self.schema_constraints = None  # Explicitly None for backward compatibility
        
        # Load schema constraints if available
        if self.schema_integrator:
            try:
                self.schema_constraints = self.schema_integrator.get_nas_constraints("neural_agent")
                logger.info(f"Loaded {len(self.schema_constraints)} schema constraint categories for NAS")
            except Exception as e:
                logger.warning(f" Failed to load schema constraints: {e}")
                self.schema_constraints = None
                
        self.build_default_architecture()

    def build_default_architecture(self):
        # Apply schema constraints to architecture if available
        if self.schema_integrator and self.schema_constraints and "architecture_constraints" in self.schema_constraints:
            arch_constraints = self.schema_constraints["architecture_constraints"]
            
            # Extract suggested dimensions from schema
            hidden_dim1 = arch_constraints.get("recommended_hidden_dim", 128)
            hidden_dim2 = arch_constraints.get("recommended_depth", 64)
            activation = arch_constraints.get("preferred_activation", "ReLU")
            
            logger.info(f"Building architecture with schema guidance: {hidden_dim1}->{hidden_dim2}, activation={activation}")
            
            # Build schema-guided architecture
            layers = [
                nn.Linear(self.input_dim, hidden_dim1),
                self._get_activation_layer(activation),
                nn.Linear(hidden_dim1, hidden_dim2),
                self._get_activation_layer(activation),
                nn.Linear(hidden_dim2, self.output_dim)
            ]
        else:
            # Default architecture (legacy mode)
            layers = [
                nn.Linear(self.input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.output_dim)
            ]
            
        self.layers = nn.Sequential(*layers).to(self.device)

    def _get_activation_layer(self, activation_name: str):
        """Get activation layer from name with schema compatibility"""
        activation_map = {
            "ReLU": nn.ReLU(),
            "Tanh": nn.Tanh(),
            "Sigmoid": nn.Sigmoid(),
            "LeakyReLU": nn.LeakyReLU(),
            "GELU": nn.GELU() if hasattr(nn, 'GELU') else nn.ReLU()
        }
        return activation_map.get(activation_name, nn.ReLU())

    def propose_architecture(self) -> Dict:
        """Propose a new architecture based on NAS logic with schema compliance."""
        if self.schema_integrator and self.schema_constraints and "evolution_template" in self.schema_constraints:
            # Use schema template for architecture proposal
            template = self.schema_constraints["evolution_template"]
            
            proposed_architecture = {
                'layer1': {
                    'type': 'Linear', 
                    'in_features': self.input_dim, 
                    'out_features': template.get("layer_width", 128)
                },
                'layer2': {
                    'type': template.get("activation", "ReLU")
                },
                'layer3': {
                    'type': 'Linear', 
                    'in_features': template.get("layer_width", 128), 
                    'out_features': template.get("depth_factor", 64)
                },
                'layer4': {
                    'type': template.get("activation", "ReLU")
                },
                'layer5': {
                    'type': 'Linear', 
                    'in_features': template.get("depth_factor", 64), 
                    'out_features': self.output_dim
                },
                'schema_metadata': {
                    'template_source': template.get("source_schema", "unknown"),
                    'cognitive_constraints': template.get("cognitive_requirements", []),
                    'compliance_level': template.get("compliance_target", 0.8)
                }
            }
            
            logger.info(f"Proposed schema-compliant architecture with {len(template.get('cognitive_requirements', []))} cognitive constraints")
            
        else:
            # Default proposal without schema guidance
            proposed_architecture = {
                'layer1': {'type': 'Linear', 'in_features': self.input_dim, 'out_features': 128},
                'layer2': {'type': 'ReLU'},
                'layer3': {'type': 'Linear', 'in_features': 128, 'out_features': 64},
                'layer4': {'type': 'ReLU'},
                'layer5': {'type': 'Linear', 'in_features': 64, 'out_features': self.output_dim}
            }
            
        return proposed_architecture
    
    def validate_architecture_compliance(self, architecture: Dict) -> Dict:
        """Validate proposed architecture against schema constraints (backward compatible)"""
        if not self.schema_integrator or not self.schema_constraints:
            return {"compliant": True, "score": 1.0, "violations": []}
            
        try:
            # Use schema integrator to validate architecture
            validation_result = self.schema_integrator.validate_evolved_architecture(
                architecture, "voxsigil_1_4_uni"
            )
            
            logger.info(f"Architecture compliance score: {validation_result.get('compliance_score', 0.0):.2f}")
            return validation_result
            
        except Exception as e:
            logger.error(f"[ERROR] Architecture validation failed: {e}")
            return {"compliant": False, "score": 0.0, "violations": [str(e)]}

    def get_schema_guided_mutations(self) -> Dict:
        """Get mutation strategies based on schema constraints"""
        if not self.schema_constraints:
            return {"strategies": ["random"], "mutation_rate": 0.2}
            
        constraints = self.schema_constraints.get("mutation_constraints", {})
        return {
            "strategies": constraints.get("allowed_mutations", ["weight_perturbation", "architecture_modification"]),
            "mutation_rate": constraints.get("mutation_rate", 0.15),
            "constraint_preservation": constraints.get("preserve_constraints", True),
            "cognitive_coherence": constraints.get("maintain_coherence", True)
        }

# EVO: Evolutionary Optimization with Schema-Driven Evolution
class EvolutionaryOptimizer:
    def __init__(self, input_dim, output_dim, training_data, validation_data, 
                 nas_architecture=None, population_size=10, device=None, schema_integrator=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.training_data = training_data
        self.validation_data = validation_data
        self.nas_architecture = nas_architecture
        self.population_size = population_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.schema_integrator = schema_integrator
        self.evolution_constraints = None  # Explicitly None for backward compatibility
        
        # Load evolution constraints from schemas
        if self.schema_integrator:
            try:
                self.evolution_constraints = self.schema_integrator.get_evo_constraints("neural_evolution")
                logger.info(f"Loaded {len(self.evolution_constraints)} evolution constraint categories")
            except Exception as e:
                logger.warning(f"[WARNING] Failed to load evolution constraints: {e}")
                self.evolution_constraints = None

        # DEAP toolbox setup with schema-aware methods
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.initialize_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self.schema_aware_reproduce)
        self.toolbox.register("mutate", self.schema_aware_mutate)
        self.toolbox.register("select", tools.selBest)
        self.toolbox.register("evaluate", self.schema_aware_evaluate)

    def initialize_individual(self) -> creator.Individual:
        """Initialize an individual using NAS architecture or defaults with schema guidance."""
        weights = []
        
        # Apply schema initialization constraints if available
        if self.schema_integrator and self.evolution_constraints and "initialization" in self.evolution_constraints:
            init_constraints = self.evolution_constraints.get("initialization", {})
            weight_init_std = init_constraints.get("weight_std", 0.1)
            init_method = init_constraints.get("method", "normal")
            
            logger.debug(f"Initializing individual with schema method: {init_method}, std: {weight_init_std}")
        else:
            weight_init_std = 1.0
            init_method = "normal"
        
        if self.nas_architecture:
            for layer in self.nas_architecture.values():
                if isinstance(layer, dict) and layer.get('type') == 'Linear':
                    if init_method == "xavier":
                        weight = torch.empty(layer['out_features'], layer['in_features'], device=self.device)
                        nn.init.xavier_normal_(weight)
                    else:
                        weight = torch.randn(layer['out_features'], layer['in_features'], device=self.device) * weight_init_std
                    weights.append(weight)
        else:
            # Default weights with schema-guided initialization
            if init_method == "xavier":
                w1 = torch.empty(128, self.input_dim, device=self.device)
                w2 = torch.empty(64, 128, device=self.device)
                w3 = torch.empty(self.output_dim, 64, device=self.device)
                nn.init.xavier_normal_(w1)
                nn.init.xavier_normal_(w2)
                nn.init.xavier_normal_(w3)
                weights = [w1, w2, w3]
            else:
                weights = [
                    torch.randn(128, self.input_dim, device=self.device) * weight_init_std,
                    torch.randn(64, 128, device=self.device) * weight_init_std,
                    torch.randn(self.output_dim, 64, device=self.device) * weight_init_std
                ]
                
        individual = creator.Individual(weights)
        
        # Add schema metadata to individual
        if hasattr(individual, '__dict__'):
            individual.schema_metadata = {
                "initialization_method": init_method,
                "constraint_compliance": self._check_individual_compliance(individual),
                "generation": 0,
                "mutation_history": []
            }
            
        return individual

    def _check_individual_compliance(self, individual) -> float:
        """Check how well an individual complies with schema constraints"""
        if not self.evolution_constraints:
            return 1.0  # Perfect compliance when no constraints (backward compatibility)
            
        try:
            # Convert individual to architecture format for validation
            arch_dict = self._individual_to_architecture(individual)
            
            if self.schema_integrator:
                validation = self.schema_integrator.validate_evolved_architecture(arch_dict, "voxsigil_1_4_uni")
                return validation.get("compliance_score", 0.0)
            else:
                return 1.0
                
        except Exception as e:
            logger.debug(f"Compliance check failed: {e}")
            return 0.5  # Neutral score if validation fails
    
    def _individual_to_architecture(self, individual) -> Dict:
        """Convert individual weights to architecture dictionary for validation"""
        arch_dict = {
            "layers": [],
            "weights": [w.shape for w in individual],
            "complexity": sum(w.numel() for w in individual),
            "device": str(self.device)
        }
        
        for i, weight in enumerate(individual):
            arch_dict["layers"].append({
                "layer_id": i,
                "type": "Linear",
                "shape": list(weight.shape),
                "parameters": weight.numel()
            })
            
        return arch_dict

    def schema_aware_evaluate(self, individual) -> Tuple[float, float]:
        """Evaluate individual with schema compliance as additional fitness dimension."""
        # Standard performance evaluation
        model = self.build_model(individual).to(self.device)
        train_loader = DataLoader(TensorDataset(*self.training_data), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(*self.validation_data), batch_size=32, shuffle=False)
        optimizer = Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        # Train for epochs (with schema-driven or default settings)
        if self.schema_integrator and self.evolution_constraints:
            epochs = self.evolution_constraints.get("training", {}).get("epochs_per_eval", 3)
        else:
            epochs = 3  # Default for backward compatibility
            
        model.train()
        for _ in range(epochs):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        val_loss /= len(val_loader)

        # Schema compliance score (only when schema is available)
        if self.schema_integrator and self.evolution_constraints:
            compliance_score = self._check_individual_compliance(individual)
            schema_fitness = -compliance_score  # Negate because DEAP minimizes by default
            schema_weight = self.evolution_constraints.get("fitness", {}).get("schema_weight", 0.3)
            combined_loss = val_loss + schema_weight * schema_fitness
        else:
            combined_loss = val_loss  # No schema penalty in legacy mode
        
        # Complexity penalty
        complexity = sum(torch.norm(w).item() for w in individual)
        
        return combined_loss, complexity

    def schema_aware_reproduce(self, ind1, ind2):
        """Schema-aware crossover between two individuals."""
        if self.schema_integrator and self.evolution_constraints:
            crossover_constraints = self.evolution_constraints.get("crossover", {})
            crossover_rate = crossover_constraints.get("rate", 0.5)
            preserve_structure = crossover_constraints.get("preserve_structure", True)
        else:
            # Default crossover settings for backward compatibility
            crossover_rate = 0.5
            preserve_structure = True
        
        if preserve_structure:
            # Structured crossover that maintains layer coherence
            for i in range(len(ind1)):
                if random.random() < crossover_rate:
                    # Swap entire layers rather than individual weights
                    ind1[i], ind2[i] = ind2[i].clone(), ind1[i].clone()
        else:
            # Standard element-wise crossover
            for i in range(len(ind1)):
                if random.random() < crossover_rate:
                    ind1[i], ind2[i] = ind2[i], ind1[i]
                    
        # Update schema metadata (if available)
        for ind in [ind1, ind2]:
            if hasattr(ind, 'schema_metadata') and self.schema_integrator:
                ind.schema_metadata["mutation_history"].append("crossover")
                ind.schema_metadata["constraint_compliance"] = self._check_individual_compliance(ind)
                
        return ind1, ind2

    def schema_aware_mutate(self, individual):
        """Schema-guided mutation of an individual."""
        if self.schema_integrator and self.evolution_constraints:
            mutation_constraints = self.evolution_constraints.get("mutation", {})
            mutation_rate = mutation_constraints.get("rate", 0.2)
            mutation_strength = mutation_constraints.get("strength", 0.1)
            adaptive_strength = mutation_constraints.get("adaptive", False)
        else:
            # Default mutation settings for backward compatibility
            mutation_rate = 0.2
            mutation_strength = 0.1
            adaptive_strength = False
        
        # Adaptive mutation strength based on compliance
        if adaptive_strength and hasattr(individual, 'schema_metadata'):
            compliance = individual.schema_metadata.get("constraint_compliance", 0.5)
            # Higher compliance = smaller mutations (fine-tuning)
            # Lower compliance = larger mutations (exploration)
            mutation_strength *= (1.0 - compliance + 0.1)
        
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                # Schema-guided mutation noise
                mutation_noise = torch.randn_like(individual[i]) * mutation_strength
                individual[i] += mutation_noise
                
        # Update schema metadata (if available)
        if hasattr(individual, 'schema_metadata') and self.schema_integrator:
            individual.schema_metadata["mutation_history"].append("point_mutation")
            individual.schema_metadata["constraint_compliance"] = self._check_individual_compliance(individual)
            
        return individual,

    def build_model(self, individual: creator.Individual) -> nn.Module:
        """Build a PyTorch model from an individual's weights with schema compliance."""
        # Check if we have schema-guided architecture
        if self.nas_architecture and "schema_metadata" in self.nas_architecture:
            # Build model according to schema guidance
            layers = []
            layer_idx = 0
            
            for layer_name, layer_config in self.nas_architecture.items():
                if layer_name == "schema_metadata":
                    continue
                    
                if isinstance(layer_config, dict) and layer_config.get('type') == 'Linear':
                    layers.append(nn.Linear(layer_config['in_features'], layer_config['out_features']))
                    if layer_idx < len(individual):
                        layers[-1].weight.data = individual[layer_idx].clone()
                        layer_idx += 1
                elif isinstance(layer_config, dict) and 'type' in layer_config:
                    # Add activation layers based on schema
                    activation_type = layer_config['type']
                    if activation_type == 'ReLU':
                        layers.append(nn.ReLU())
                    elif activation_type == 'Tanh':
                        layers.append(nn.Tanh())
                    elif activation_type == 'Sigmoid':
                        layers.append(nn.Sigmoid())
                    # Add more as needed
        else:
            # Default model structure
            layers = []
            layers.append(nn.Linear(self.input_dim, 128))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(128, 64))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(64, self.output_dim))
            
            # Apply weights from the individual
            linear_layer_idx = 0
            for layer in layers:
                if isinstance(layer, nn.Linear) and linear_layer_idx < len(individual):
                    layer.weight.data = individual[linear_layer_idx].clone()
                    linear_layer_idx += 1
                    
        model = nn.Sequential(*layers)
        return model

    def reproduce(self, ind1, ind2):
        """Legacy crossover method (kept for compatibility)."""
        return self.schema_aware_reproduce(ind1, ind2)

    def mutate(self, individual):
        """Legacy mutation method (kept for compatibility)."""
        return self.schema_aware_mutate(individual)

    def evolve(self, generations=10):
        """Run schema-guided evolutionary optimization."""
        logger.info(f"Starting schema-guided evolution for {generations} generations")
        
        population = self.toolbox.population(n=self.population_size)
        
        # Track schema compliance over generations
        compliance_history = []
        best_fitness_history = []

        for gen in range(generations):
            logger.info(f"Generation {gen + 1}/{generations}")
            
            # Evaluate all individuals
            fitnesses = list(map(self.toolbox.evaluate, population))

            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # Track generation statistics
            current_compliances = [
                getattr(ind, 'schema_metadata', {}).get('constraint_compliance', 0.0) 
                for ind in population
            ]
            avg_compliance = sum(current_compliances) / len(current_compliances) if current_compliances else 0.0
            compliance_history.append(avg_compliance)
            
            best_fitness = min(fit[0] for fit in fitnesses)
            best_fitness_history.append(best_fitness)
            
            logger.info(f"Gen {gen+1}: Avg compliance: {avg_compliance:.3f}, Best fitness: {best_fitness:.4f}")

            # Selection and reproduction
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover
            crossover_rate = 0.5  # Default value
            if self.evolution_constraints is not None:
                crossover_rate = self.evolution_constraints.get("crossover", {}).get("global_rate", 0.5)
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < crossover_rate:
                    self.toolbox.mate(child1, child2)

            # Apply mutation
            mutation_rate = 0.2  # Default value
            if self.evolution_constraints is not None:
                mutation_rate = self.evolution_constraints.get("mutation", {}).get("global_rate", 0.2)
            for mutant in offspring:
                if random.random() < mutation_rate:
                    self.toolbox.mutate(mutant)

            population[:] = offspring

        # Final evaluation and reporting
        final_fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, final_fitnesses):
            ind.fitness.values = fit
            
        best_individuals = tools.selBest(population, k=3)
        
        logger.info("Schema-Guided Evolution Results:")
        for i, ind in enumerate(best_individuals):
            compliance = getattr(ind, 'schema_metadata', {}).get('constraint_compliance', 'unknown')
            logger.info(f"Best individual {i + 1}: fitness={ind.fitness.values}, compliance={compliance}")
            
        # Return evolution results
        return {
            "best_individuals": best_individuals,
            "compliance_history": compliance_history,
            "fitness_history": best_fitness_history,
            "final_population": population,
            "schema_constraints_used": len(self.evolution_constraints) if self.evolution_constraints is not None else 0,
            "generations_completed": generations
        }
    
    async def evolve_architecture(self, current_architecture=None, target_compliance=0.85, generations=5):
        """Evolve architecture to improve schema compliance - async wrapper for evolve method"""
        import asyncio
        
        logger.info(f"🧬 Evolving architecture with target compliance {target_compliance:.3f}")
        
        # Run evolution in a separate thread to maintain async compatibility
        def run_evolution():
            try:
                results = self.evolve(generations=generations)
                
                if results and "best_individuals" in results:
                    best_individual = results["best_individuals"][0]
                    
                    # Convert best individual to architecture specification
                    architecture_spec = self._individual_to_architecture(best_individual)
                    
                    # Add evolution metadata
                    architecture_spec["evolution_metadata"] = {
                        "fitness": best_individual.fitness.values if hasattr(best_individual, 'fitness') else [0.0, 0.0],
                        "compliance": getattr(best_individual, 'schema_metadata', {}).get('constraint_compliance', 0.0),
                        "generations": generations,
                        "population_size": self.population_size,
                        "selection_pressure": results.get("schema_constraints_used", 0) / max(1, len(results.get("final_population", [1]))),
                        "fitness_delta": (results["fitness_history"][0] - results["fitness_history"][-1]) if results["fitness_history"] else 0.0,
                        "diversity_score": len(set(str(ind) for ind in results["best_individuals"])) / len(results["best_individuals"]) if results["best_individuals"] else 0.0
                    }
                    
                    logger.info(f"🎯 Evolution completed: fitness={architecture_spec['evolution_metadata']['fitness']}, compliance={architecture_spec['evolution_metadata']['compliance']:.3f}")
                    return architecture_spec
                    
            except Exception as e:
                logger.error(f"Evolution failed: {e}")
                return None
        
        # Run evolution in thread pool to maintain async behavior
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_evolution)
        return result

    def get_schema_guided_mutations(self) -> Dict:
        """Get mutation strategies based on evolution constraints"""
        if not self.evolution_constraints:
            return {"strategies": ["random"], "mutation_rate": 0.2}
            
        constraints = self.evolution_constraints.get("mutation", {})
        return {
            "strategies": constraints.get("allowed_mutations", ["weight_perturbation"]),
            "mutation_rate": constraints.get("rate", 0.2),
            "constraint_preservation": constraints.get("preserve_constraints", True)
        }

    def get_schema_guided_crossover(self) -> Dict:
        """Get crossover strategies based on evolution constraints"""
        if not self.evolution_constraints:
            return {"method": "uniform", "rate": 0.5}
            
        constraints = self.evolution_constraints.get("crossover", {})
        return {
            "method": constraints.get("method", "uniform"),
            "rate": constraints.get("rate", 0.5),
            "preserve_structure": constraints.get("preserve_structure", True)
        }

# Main Schema-Integrated EVO+NAS System
def create_schema_driven_evo_nas_system(input_dim: int, output_dim: int, 
                                       training_data: Tuple = None, validation_data: Tuple = None,
                                       device=None, use_schemas: bool = True, enable_schema_integration: bool = None):
    """Create integrated EVO+NAS system with schema-driven evolution"""
    
    # Handle both parameter names for backward compatibility
    if enable_schema_integration is not None:
        use_schemas = enable_schema_integration
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize schema integrator if available and requested
    schema_integrator = None
    if use_schemas and SCHEMA_AVAILABLE:
        try:
            from training.schema_driven_evolution import SchemaEvolutionIntegrator
            schema_integrator = SchemaEvolutionIntegrator()
            logger.info("Schema-driven evolution system initialized")
        except Exception as e:
            logger.warning(f"[WARNING] Failed to initialize schema system: {e}")
            
    # Initialize NAS with schema support
    nas = NeuralArchitectureSearch(
        input_dim=input_dim, 
        output_dim=output_dim, 
        device=device,
        schema_integrator=schema_integrator
    )
    
    # Get schema-compliant architecture
    nas_architecture = nas.propose_architecture()
    
    # Validate architecture compliance
    if schema_integrator:
        compliance_check = nas.validate_architecture_compliance(nas_architecture)
        logger.info(f"Architecture compliance: {compliance_check.get('score', 0.0):.2f}")
        
        # Enhance architecture with schema metadata if compliant
        if compliance_check.get('compliant', False):
            nas_architecture['validated'] = True
            nas_architecture['compliance_data'] = compliance_check
        
    logger.info(f"NAS Architecture: {len(nas_architecture)} layers")
    
    # Initialize EVO with schema-driven evolution
    if training_data is not None and validation_data is not None:
        evo = EvolutionaryOptimizer(
            input_dim=input_dim,
            output_dim=output_dim, 
            training_data=training_data,
            validation_data=validation_data,
            nas_architecture=nas_architecture,
            device=device,
            schema_integrator=schema_integrator
        )
    else:
        # Create dummy data for testing
        dummy_x = torch.randn(10, input_dim)
        dummy_y = torch.randn(10, output_dim)
        evo = EvolutionaryOptimizer(
            input_dim=input_dim,
            output_dim=output_dim, 
            training_data=(dummy_x, dummy_y),
            validation_data=(dummy_x, dummy_y),
            nas_architecture=nas_architecture,
            device=device,
            schema_integrator=schema_integrator
        )
    
    return nas, evo

# Main integration with schema-driven capabilities
if __name__ == "__main__":
    logger.info("Starting Schema-Driven NAS + EVO Integration")

    # System parameters
    input_dim = 10
    output_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate synthetic data
    train_data = (torch.randn(100, input_dim), torch.randn(100, output_dim))
    val_data = (torch.randn(20, input_dim), torch.randn(20, output_dim))

    # Create schema-driven system
    nas, evo = create_schema_driven_evo_nas_system(
        input_dim=input_dim,
        output_dim=output_dim, 
        training_data=train_data,
        validation_data=val_data,
        device=device,
        use_schemas=True
    )
    
    logger.info("NAS and EVO systems initialized successfully")
    
    # Run schema-guided evolution
    evolution_results = evo.evolve(generations=10)
    
    # Display results
    logger.info("Evolution Results Summary:")
    logger.info(f"   Generations completed: {evolution_results['generations_completed']}")
    logger.info(f"   Schema constraints used: {evolution_results['schema_constraints_used']}")
    logger.info(f"   Best individual fitness: {evolution_results['best_individuals'][0].fitness.values}")
    
    # Final compliance check
    schema_integrator = getattr(nas, 'schema_integrator', None)
    if schema_integrator:
        final_compliance = [
            getattr(ind, 'schema_metadata', {}).get('constraint_compliance', 0.0)
            for ind in evolution_results['best_individuals']
        ]
        avg_final_compliance = sum(final_compliance) / len(final_compliance) if final_compliance else 0.0
        logger.info(f"   Final avg compliance: {avg_final_compliance:.3f}")
        
    # Test agent creation pipeline
    if schema_integrator:
        logger.info("Testing agent creation pipeline...")
        
        # Create a test agent using the best evolved individual
        best_individual = evolution_results['best_individuals'][0]
        
        agent_requirements = {
            "type": "evolved_neural_agent",
            "description": "Agent created from schema-guided evolution",
            "evolved_weights": [w.cpu().numpy().tolist() for w in best_individual],
            "fitness_scores": best_individual.fitness.values,
            "compliance_score": getattr(best_individual, 'schema_metadata', {}).get('constraint_compliance', 0.0)
        }
        
        # This would typically be called from the BLT system
        try:
            test_agent = schema_integrator.generate_agent_from_evolution(agent_requirements)
            logger.info(f"[OK] Successfully created agent with {len(test_agent)} components")
        except Exception as e:
            logger.error(f"[ERROR] Agent creation test failed: {e}")

    logger.info("Schema-Driven NAS + EVO Integration Completed")