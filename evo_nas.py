import os
import logging
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from deap import base, creator, tools
from typing import List, Dict, Tuple, Optional
from torch.utils.tensorboard import SummaryWriter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# TensorBoard writer
writer = SummaryWriter(log_dir='runs/nas_evo')

# Define DEAP creator types
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# NAS: Neural Architecture Search
class NeuralArchitectureSearch(nn.Module):
    def __init__(self, input_dim, output_dim, device=None):
        super(NeuralArchitectureSearch, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = []
        self.build_default_architecture()

    def build_default_architecture(self):
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
        ).to(self.device)

    def propose_architecture(self) -> Dict:
        """Propose a new architecture based on NAS logic."""
        proposed_architecture = {
            'layer1': {'type': 'Linear', 'in_features': self.input_dim, 'out_features': 128},
            'layer2': {'type': 'ReLU'},
            'layer3': {'type': 'Linear', 'in_features': 128, 'out_features': 64},
            'layer4': {'type': 'ReLU'},
            'layer5': {'type': 'Linear', 'in_features': 64, 'out_features': self.output_dim}
        }
        return proposed_architecture

# EVO: Evolutionary Optimization
class EvolutionaryOptimizer:
    def __init__(self, input_dim, output_dim, training_data, validation_data, nas_architecture=None, population_size=10, device=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.training_data = training_data
        self.validation_data = validation_data
        self.nas_architecture = nas_architecture
        self.population_size = population_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # DEAP toolbox setup
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.initialize_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self.reproduce)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", tools.selBest)
        self.toolbox.register("evaluate", self.evaluate_individual)

    def initialize_individual(self) -> creator.Individual:
        """Initialize an individual using NAS architecture or defaults."""
        weights = []
        if self.nas_architecture:
            for layer in self.nas_architecture.values():
                if layer['type'] == 'Linear':
                    weights.append(torch.randn(layer['out_features'], layer['in_features'], device=self.device))
        else:
            weights = [
                torch.randn(128, self.input_dim, device=self.device),
                torch.randn(64, 128, device=self.device),
                torch.randn(self.output_dim, 64, device=self.device)
            ]
        return creator.Individual(weights)

    def evaluate_individual(self, individual) -> Tuple[float, float]:
        """Evaluate an individual on training and validation data."""
        # Build model from weights
        model = self.build_model(individual).to(self.device)
        train_loader = DataLoader(TensorDataset(*self.training_data), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(*self.validation_data), batch_size=32, shuffle=False)
        optimizer = Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        # Train for 3 epochs
        model.train()
        for _ in range(3):
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

        # Complexity as a fitness measure
        complexity = sum(torch.norm(w).item() for w in individual)
        return val_loss, complexity

    def build_model(self, individual: creator.Individual) -> nn.Module:
        """Build a PyTorch model from an individual's weights."""
        layers = []
        layers.append(nn.Linear(self.input_dim, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 64))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(64, self.output_dim))
        model = nn.Sequential(*layers)

        # Apply weights from the individual
        for i, layer in enumerate(model):
            if isinstance(layer, nn.Linear):
                layer.weight.data = individual[i].clone()
        return model

    def reproduce(self, ind1, ind2):
        """Crossover between two individuals."""
        for i in range(len(ind1)):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
        return ind1, ind2

    def mutate(self, individual):
        """Mutate an individual."""
        for i in range(len(individual)):
            if random.random() < 0.2:
                individual[i] += torch.randn_like(individual[i]) * 0.1
        return individual

    def evolve(self, generations=10):
        """Run the evolutionary optimization."""
        population = self.toolbox.population(n=self.population_size)

        for gen in range(generations):
            logger.info(f"Generation {gen + 1}/{generations}")
            fitnesses = list(map(self.toolbox.evaluate, population))

            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # Select and create the next generation
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    self.toolbox.mate(child1, child2)

            for mutant in offspring:
                if random.random() < 0.2:
                    self.toolbox.mutate(mutant)

            population[:] = offspring

        best_individuals = tools.selBest(population, k=3)
        for i, ind in enumerate(best_individuals):
            logger.info(f"Best individual {i + 1} fitness: {ind.fitness.values}")

# Main integration
if __name__ == "__main__":
    logger.info("Starting NAS + EVO Integration")

    input_dim = 10
    output_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate synthetic data
    train_data = (torch.randn(100, input_dim), torch.randn(100, output_dim))
    val_data = (torch.randn(20, input_dim), torch.randn(20, output_dim))

    # Neural Architecture Search
    nas = NeuralArchitectureSearch(input_dim, output_dim, device=device)
    nas_architecture = nas.propose_architecture()
    logger.info(f"NAS Proposed Architecture: {nas_architecture}")

    # Evolutionary Optimization
    evo = EvolutionaryOptimizer(input_dim, output_dim, train_data, val_data, nas_architecture=nas_architecture, device=device)
    evo.evolve(generations=10)

    logger.info("NAS + EVO Integration Completed")