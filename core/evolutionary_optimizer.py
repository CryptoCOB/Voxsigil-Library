import os
import logging
import random
import traceback
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
from multiprocessing import set_start_method

import multiprocessing as mp
from typing import Dict, List, Optional, Tuple, Any
import datetime

# HOLO-1.5 imports
from .base import BaseCore, vanta_core_module, CognitiveMeshRole

# Set the start method to 'spawn' for multiprocessing
set_start_method('spawn', force=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set the environment variable for CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

from deap import creator, base, tools

# Define the creator for the DEAP evolutionary algorithm
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Create directories if they don't exist
os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

def evaluate_individual_wrapper(args: Tuple[Any, ...]) -> Tuple[float, float]:
    """Wrapper function for individual evaluation to be used with multiprocessing."""
    try:
        return evaluate_individual(*args)
    except Exception as e:
        logger.error(f"Failed to evaluate individual due to: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Return a high fitness value to penalize this individual or handle as appropriate for your system
        return float('inf'), float('inf')  # Assuming lower fitness is better


def evaluate_individual(self, individual: List[torch.Tensor], 
                        input_dim: int, 
                        output_dim: int, 
                        training_data: Tuple[torch.Tensor, torch.Tensor], 
                        validation_data: Tuple[torch.Tensor, torch.Tensor], 
                        device: torch.device,
                        nas_architecture: Dict) -> Tuple[float, float]:
    """Evaluate an individual's fitness."""
    logger.debug(f"Evaluating individual with input_dim: {input_dim}, output_dim: {output_dim}")
    
    if nas_architecture:
        logger.debug(f"Using NAS architecture: {nas_architecture}")
        model = self.build_model_from_nas(nas_architecture).to(device)
    else:
        logger.debug("Using default model architecture")
        model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        ).to(device)
    
    logger.debug(f"Model architecture: {model}")

    def adjust_weights(layer: nn.Linear, individual_layer: torch.Tensor) -> torch.Tensor:
        layer_shape = layer.weight.data.shape
        ind_shape = individual_layer.shape
        logger.debug(f"Adjusting weights. Layer shape: {layer_shape}, Individual shape: {ind_shape}")
        if layer_shape != ind_shape:
            logger.warning(f"Shape mismatch. Layer: {layer_shape}, Individual: {ind_shape}")
            padded_layer = torch.zeros(layer_shape, device=device)
            padded_layer[:min(layer_shape[0], ind_shape[0]), :min(layer_shape[1], ind_shape[1])] = \
                individual_layer[:min(layer_shape[0], ind_shape[0]), :min(layer_shape[1], ind_shape[1])]
            logger.debug(f"Padded layer shape: {padded_layer.shape}")
            return padded_layer
        return individual_layer

    # Apply weights from individual to model
    linear_layers = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
    for i, (layer, ind_weight) in enumerate(zip(linear_layers, individual)):
        if layer.weight.data.shape == ind_weight.shape:
            layer.weight.data = ind_weight.clone().detach().to(device)
        else:
            logger.warning(f"Shape mismatch in layer {i}. Layer: {layer.weight.data.shape}, Individual: {ind_weight.shape}")
            layer.weight.data = adjust_weights(layer, ind_weight).to(device)

    # Training and evaluation logic
    train_texts, train_labels = training_data
    val_texts, val_labels = validation_data

    logger.debug(f"Training data shapes - Inputs: {train_texts.shape}, Labels: {train_labels.shape}")
    logger.debug(f"Validation data shapes - Inputs: {val_texts.shape}, Labels: {val_labels.shape}")

    train_data = TensorDataset(train_texts.clone().detach().float(), train_labels.clone().detach().float())
    val_data = TensorDataset(val_texts.clone().detach().float(), val_labels.clone().detach().float())

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    # Using Huggingface's Accelerator for distributed training across GPUs
    accelerator = Accelerator()
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    model.train()
    for epoch in range(5):  # Reduced number of epochs for faster evaluation
        train_loss = 0
        for inputs, labels in train_loader:
            torch.cuda.empty_cache()  # Clear cache to manage GPU memory
            optimizer.zero_grad()
            outputs = model(inputs)
            logger.debug(f"Batch - Input shape: {inputs.shape}, Output shape: {outputs.shape}, Label shape: {labels.shape}")
            loss = criterion(outputs, labels)
            accelerator.backward(loss)
            optimizer.step()
            train_loss += loss.item()
        logger.debug(f"Epoch {epoch+1}/5, Train Loss: {train_loss/len(train_loader)}")

    # Evaluate on validation set
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    fitness = val_loss / len(val_loader)
    complexity = torch.mean(torch.stack([torch.mean(layer ** 2) for layer in individual])).item()
    logger.debug(f"Individual evaluation complete. Fitness: {fitness}, Complexity: {complexity}")
    return fitness, complexity


def build_model_from_nas(self, nas_architecture: Optional[Dict]) -> nn.Module:
    """
    Build the model based on the NAS architecture.

    Args:
        nas_architecture: Dictionary defining the NAS architecture.

    Returns:
        An nn.Sequential model based on the provided NAS architecture.
    """
    if nas_architecture is None:
        # If no NAS architecture is provided, use a default architecture
        self.logger.info("No NAS architecture provided. Using default architecture.")
        return nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
        )
    
    layers = []
    
    # Iterate over the layers defined in the NAS architecture
    for layer_name, layer_info in nas_architecture.items():
        layer_type = layer_info.get('type', None)
        
        if layer_type == 'Linear':
            # Add Linear layer
            layers.append(nn.Linear(layer_info['in_features'], layer_info['out_features']))
            self.logger.debug(f"Added Linear layer: {layer_info['in_features']} -> {layer_info['out_features']}")
        
        elif layer_type == 'ReLU':
            # Add ReLU activation
            layers.append(nn.ReLU())
            self.logger.debug(f"Added ReLU activation.")
        
        elif layer_type == 'Dropout':
            # Add Dropout layer
            p = layer_info.get('p', 0.5)  # Default dropout probability is 0.5
            layers.append(nn.Dropout(p))
            self.logger.debug(f"Added Dropout with p={p}.")
        
        elif layer_type == 'BatchNorm':
            # Add Batch Normalization layer
            layers.append(nn.BatchNorm1d(layer_info['num_features']))
            self.logger.debug(f"Added BatchNorm1d with num_features={layer_info['num_features']}.")
        
        elif layer_type == 'Conv2d':
            # Add 2D Convolution layer if architecture specifies
            layers.append(nn.Conv2d(layer_info['in_channels'], layer_info['out_channels'], kernel_size=layer_info['kernel_size']))
            self.logger.debug(f"Added Conv2d layer: {layer_info['in_channels']} -> {layer_info['out_channels']} with kernel_size={layer_info['kernel_size']}")

        elif layer_type == 'MaxPool2d':
            # Add MaxPooling layer if required
            layers.append(nn.MaxPool2d(kernel_size=layer_info.get('kernel_size', 2)))
            self.logger.debug(f"Added MaxPool2d with kernel_size={layer_info.get('kernel_size', 2)}.")

        elif layer_type == 'Flatten':
            # Add a Flatten layer if necessary
            layers.append(nn.Flatten())
            self.logger.debug(f"Added Flatten layer.")

        else:
            # Log a warning if the layer type is not supported
            self.logger.warning(f"Unknown layer type in NAS architecture: {layer_type}")

    # Return the constructed model
    self.logger.info("Successfully built model from NAS architecture.")
    return nn.Sequential(*layers)


@vanta_core_module(
    name="evolutionary_optimizer",
    subsystem="optimization",
    mesh_role=CognitiveMeshRole.SYNTHESIZER,
    description="Evolutionary optimization system with multi-GPU support and NAS integration",
    capabilities=["evolutionary_algorithms", "multi_gpu_training", "population_optimization", "fitness_evaluation", "genetic_operations"],
    cognitive_load=4.0,
    symbolic_depth=4,
    collaboration_patterns=["evolutionary_synthesis", "distributed_optimization", "genetic_collaboration"]
)
class EvolutionaryOptimizer(BaseCore):
    def __init__(self, vanta_core, config: Optional[Dict[str, Any]] = None, 
                 input_dim: int = 768, output_dim: int = 768, 
                 training_data: Tuple[torch.Tensor, torch.Tensor] = None, 
                 validation_data: Tuple[torch.Tensor, torch.Tensor] = None, 
                 device: torch.device = None, 
                 population_size: int = 50, mutation_rate_decay: float = 0.95, 
                 mutation_rate_start: float = 0.5, 
                 mutation_rate: float = 0.03, nas_architecture: Optional[Dict] = None, 
                 use_multiprocessing: bool = True, **kwargs):
        
        # Initialize BaseCore
        super().__init__(vanta_core, config or {})
        
        # Use provided use_multiprocessing argument or default to True
        self.use_multiprocessing = use_multiprocessing
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing EvolutionaryOptimizer with HOLO-1.5 enhancement")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.training_data = training_data
        self.validation_data = validation_data
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.population_size = population_size
        self.mutate_rate = mutation_rate
        self.mutation_rate_start = mutation_rate_start
        self.mutation_rate_decay = mutation_rate_decay
        self.nas_architecture = nas_architecture

        self.logger.info(f"Input dimension: {input_dim}, Output dimension: {output_dim}")
        self.logger.info(f"NAS architecture: {nas_architecture}")

        # Handle training data validation if provided
        if training_data:
            _, labels = training_data
            if len(labels.shape) == 1:
                # Handle 1D labels scenario, such as treating it as a single-class problem
                if output_dim != 1:
                    self.logger.error(f"Expected output_dim of 1 for 1D labels, got {output_dim}")
                    raise ValueError(f"Output dimension mismatch. Expected 1D labels for output_dim=1.")
            elif labels.shape[1] != output_dim:
                self.logger.warning(f"Output dimension mismatch detected. Expected: {output_dim}, Got: {labels.shape[1]}. Attempting to transpose labels.")
                # Attempt to transpose the labels
                labels = labels.T
                if labels.shape[0] == output_dim:
                    self.logger.info("Successfully transposed labels to match the expected output dimension.")
                else:
                    self.logger.error(f"Unable to resolve dimension mismatch. Expected: {output_dim}, Got after transpose: {labels.shape[1]}")
                    raise ValueError(f"Output dimension mismatch. Check your data and configuration.")

        if self.use_multiprocessing:
            self.num_processes = torch.cuda.device_count()
        else:
            self.num_processes = os.cpu_count()

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.initialize_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self.reproduce)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("evaluate", evaluate_individual_wrapper)

        try:
            if training_data and validation_data:
                self.population = self.initialize_population()
                self.writer = SummaryWriter(log_dir='runs/evolutionary_optimizer')
                self.fitness_scores = self.evaluate_population(self.population)
        except Exception as e:
            self.logger.error(f"Error during initialization: {e}")
            raise
            
        self.elite_weights = []
        self.gen_handler = GenerationHandler(self)

    async def initialize(self) -> bool:
        """Initialize the EvolutionaryOptimizer for BaseCore compliance."""
        try:
            self.logger.info("EvolutionaryOptimizer initialized successfully with HOLO-1.5 enhancement")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing EvolutionaryOptimizer: {e}")
            return False

    def setup_logging(self) -> logging.Logger:
        """Set up logging for the optimizer."""
        logger = logging.getLogger('EvolutionaryOptimizer')
        handler = logging.FileHandler('logs/evo_optimizer.log')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        return logger

    # ...existing code... (all the rest of your methods remain the same)
    
    async def optimize(self, parameters):
        """
        Optimize the parameters of the meta-learner using an evolutionary algorithm.

        Args:
            parameters (iterable): The parameters of the meta-learner.

        Returns:
            dict: The optimized proposal.
        """
        try:
            self.logger.info("Starting optimization with Evolutionary Optimizer.")

            # Evolve the population for a defined number of generations
            num_generations = 10  # You can adjust this value based on your needs
            for generation in range(num_generations):
                self.logger.info(f"Generation {generation + 1}/{num_generations}")
                
                # Evolve the population
                offspring = await self.evolve_population()

                # Clear cache after generating offspring
                torch.cuda.empty_cache()
                self.logger.debug("Cache cleared after generating offspring")

                # Combine current population and offspring, then select the next generation
                self.population = self.toolbox.select(self.population + offspring, self.population_size)

                # Clear cache after selection process
                torch.cuda.empty_cache()
                self.logger.debug(f"Cache cleared after population selection for generation {generation + 1}")

            # Extract the best individual
            best_individual = tools.selBest(self.population, 1)[0]
            self.logger.info("Optimization complete. Best individual selected.")

            # Log the best individual's details (optional)
            self.logger.debug(f"Best individual: {best_individual}")

            # Convert the best individual into a proposal format
            optimized_proposal = self._convert_to_proposal(best_individual)

            return optimized_proposal

        except Exception as e:
            self.logger.error(f"Error during optimization: {e}")
            raise

        
    def to(self, device: torch.device):
        """Move the optimizer and its components to the specified device."""
        self.device = device
        if hasattr(self, 'model'):
            self.model.to(device)
        # Move other relevant components to the device
        self.logger.info(f"Moved EvolutionaryOptimizer to {device}")
        return self
    
    def initialize_individual_with_nas(self) -> Any:
        """Initialize a single individual based on the NAS architecture."""
        self.logger.debug("Initializing individual with NAS architecture")

        # Fall back to default initialization if no NAS architecture is provided
        if self.nas_architecture is None:
            return self.initialize_individual()

        layers = []
        try:
            for layer_name, layer_info in self.nas_architecture.items():
                # Ensure layer_info contains the required 'in_features' and 'out_features'
                if isinstance(layer_info, dict) and 'in_features' in layer_info and 'out_features' in layer_info:
                    # Initialize the layer with random weights on the specified device
                    layers.append(torch.randn(layer_info['out_features'], layer_info['in_features'], device=self.device))
                    self.logger.debug(f"Initialized layer {layer_name}: {layer_info}")
                else:
                    # Log a warning if the expected structure is not found
                    self.logger.warning(f"Unexpected layer info in NAS architecture: {layer_name}: {layer_info}")
            
            # Clear the cache after initializing all layers
            torch.cuda.empty_cache()
            self.logger.debug("Cache cleared after NAS-based individual initialization")
            
        except Exception as e:
            self.logger.error(f"Error initializing individual with NAS: {e}", exc_info=True)
            raise

        # Log the initialized layer shapes
        self.logger.debug(f"Individual initialized with NAS-based layer shapes: {[layer.shape for layer in layers]}")
        
        # Return the individual (ensure that creator.Individual is correctly defined elsewhere)
        return creator.Individual(layers)

    # ...existing code continues with all the rest of your methods...
    
    def evaluate_population(self, population: List[Any]) -> List[Tuple[float, float]]:
        """Evaluate the entire population using all available GPUs."""
        self.logger.debug("Evaluating population")

        fitnesses = []

        # Check if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs for population evaluation")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Process each individual in the population
        for i, ind in enumerate(population):
            try:
                self.logger.debug(f"Evaluating individual {i+1}/{len(population)}")

                # Use the evaluate_individual_gpu function, which should handle multi-GPU execution
                fit = self.evaluate_individual_gpu(
                    ind, 
                    self.input_dim, 
                    self.output_dim, 
                    self.training_data, 
                    self.validation_data, 
                    device, 
                    self.nas_architecture
                )
                fitnesses.append(fit)

                # Clear the cache after evaluating each individual to free up GPU memory
                torch.cuda.empty_cache()
                self.logger.debug(f"Cache cleared after evaluating individual {i+1}")
                
            except Exception as e:
                self.logger.error(f"Error during GPU evaluation for individual {i+1}: {e}")
                raise

        # Assign fitness values back to the individuals
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        return fitnesses

    # ... continue with all your existing methods unchanged ...


# ... (Continue with all the remaining classes and methods from your original code) ...

class ContextualLogger:
    def __init__(self, logger, context):
        self.logger = logger
        self.context = context

    def debug(self, message):
        self.logger.debug(f"[{self.context}] {message}")

    def info(self, message):
        self.logger.info(f"[{self.context}] {message}")

    def warning(self, message):
        self.logger.warning(f"[{self.context}] {message}")

    def error(self, message):
        self.logger.error(f"[{self.context}] {message}")

    def critical(self, message):
        self.logger.critical(f"[{self.context}] {message}")


class GenerationHandler:
    def __init__(self, evolutionary_optimizer: EvolutionaryOptimizer):
        self.evolutionary_optimizer = evolutionary_optimizer
        self.logger = evolutionary_optimizer.logger
        self.current_generation = 0
        self.mutation_rate = evolutionary_optimizer.mutation_rate_start
        self.mutation_rate_decay = evolutionary_optimizer.mutation_rate_decay
        self.elite_folder = "elites"
        self.log_folder = "logs"
        self.setup_folders()

    def setup_folders(self):
        if not os.path.exists(self.elite_folder):
            os.makedirs(self.elite_folder)
            self.logger.info(f"Created folder: {self.elite_folder}")
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
            self.logger.info(f"Created folder: {self.log_folder}")

    def save_elites(self, elites, generation):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, elite in enumerate(elites):
            elite_path = os.path.join(self.elite_folder, f"elite_gen{generation}_rank{i+1}_{timestamp}.pth")
            with open(elite_path, 'wb') as f:
                torch.save(elite, f)
            self.logger.info(f"Saved elite {i+1} of generation {generation} to {elite_path}")

    def evolve(self, generation: int, max_generations: int):
        """Evolve the population for one generation."""
        self.logger.debug(f"Evolving generation {generation}")
        
        # Selection
        offspring = self.evolutionary_optimizer.toolbox.select(
            self.evolutionary_optimizer.population, 
            len(self.evolutionary_optimizer.population)
        )
        offspring = list(map(self.evolutionary_optimizer.toolbox.clone, offspring))
        self.logger.debug(f"Selected {len(offspring)} offspring for next generation")

        # Elite selection
        elite_size = int(0.1 * len(self.evolutionary_optimizer.population))
        elites = tools.selBest(self.evolutionary_optimizer.population, elite_size)
        self.logger.debug(f"Selected {len(elites)} elite individuals")

        # Incentive for higher-scoring elites
        if elites:
            top_elite = elites[0]
            top_elite_count = 2  # Allow the top elite to reproduce more
            for _ in range(top_elite_count):
                offspring.append(self.evolutionary_optimizer.toolbox.clone(top_elite))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if torch.rand(1).item() < 0.5:
                self.evolutionary_optimizer.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        self.logger.debug("Crossover operations completed")

        # Random possibility of twins
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < 0.1:  # 10% chance of twins
                offspring.append(self.evolutionary_optimizer.toolbox.clone(offspring[i]))
        self.logger.debug("Twins possibility applied")

        # Mutation
        mutation_rate = self.mutation_rate * (1 - generation / max_generations)
        for mutant in offspring:
            if random.random() < mutation_rate:
                self.evolutionary_optimizer.toolbox.mutate(mutant)
                del mutant.fitness.values
        self.logger.debug("Mutation operations completed")
        self.mutation_rate *= self.mutation_rate_decay

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = self.evolutionary_optimizer.evaluate_population(invalid_ind)
        self.logger.debug(f"Evaluated {len(invalid_ind)} invalid individuals")

        # Update population
        self.evolutionary_optimizer.population[:] = elites + offspring[:-elite_size]

        # Save elites
        self.save_elites(elites, generation)

        # Log progress
        best_fitness = min(ind.fitness.values[0] for ind in self.evolutionary_optimizer.population)
        self.evolutionary_optimizer.writer.add_scalar('Best Fitness', best_fitness, generation)
        self.logger.info(f"Generation {generation}: Best Fitness = {best_fitness}")

        self.current_generation += 1
        self.logger.debug(f"Generation {generation} evolution completed")
        
    def get_state(self):
        # Return the state dictionary for Evolutionary Optimizer
        return self.state_dict()

    def set_state(self, state_dict):
        # Set the state dictionary for Evolutionary Optimizer
        self.load_state_dict(state_dict)


logger.info("HOLO-1.5 Evolutionary Optimizer Infrastructure initialized")
