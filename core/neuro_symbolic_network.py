"""
NeuroSymbolic Network implementation.

This module provides a hybrid neural-symbolic network that combines the learning
capabilities of neural networks with the interpretability of symbolic reasoning.
"""

import logging
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NeuroSymbolicNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(NeuroSymbolicNetwork, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.Dropout(0.3))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.network = nn.Sequential(*layers)
        self.graph = nx.DiGraph()
        self.connected_to = []

    def forward(self, x):
        if self.training and x.size(0) == 1:
            logger.warning(
                "Switching to evaluation mode for batch normalization with batch size 1."
            )
            self.eval()
            output = self.network(x)
            self.train()
            return output
        return self.network(x)

    def add_symbolic_rule(self, rule: List[Tuple[int, int]]):
        self.graph.add_edges_from(rule)
        logger.info(f"Added symbolic rule: {rule}")

    def delete_symbolic_rule(self, rule: List[Tuple[int, int]]):
        self.graph.remove_edges_from(rule)
        logger.info(f"Deleted symbolic rule: {rule}")

    def modify_symbolic_rule(
        self, old_rule: List[Tuple[int, int]], new_rule: List[Tuple[int, int]]
    ):
        self.delete_symbolic_rule(old_rule)
        self.add_symbolic_rule(new_rule)
        logger.info(f"Modified symbolic rule from {old_rule} to {new_rule}")

    def reason(self, node):
        path = list(nx.dfs_edges(self.graph, node))
        logger.info(f"Reasoning path from node {node}: {path}")
        return path

    def reason_bfs(self, node):
        path = list(nx.bfs_edges(self.graph, node))
        logger.info(f"BFS reasoning path from node {node}: {path}")
        return path

    def shortest_path(self, source, target):
        path = nx.shortest_path(self.graph, source, target)
        logger.info(f"Shortest path from node {source} to node {target}: {path}")
        return path

    def add_weighted_rule(self, rule: List[Tuple[int, int, float]]):
        self.graph.add_weighted_edges_from(rule)
        logger.info(f"Added weighted rule: {rule}")

    def reason_with_weights(self, source, target):
        path = nx.dijkstra_path(self.graph, source, target)
        logger.info(
            f"Reasoning path with weights from node {source} to node {target}: {path}"
        )
        return path

    def visualize_graph(self):
        pos = nx.spring_layout(self.graph)
        edge_labels = nx.get_edge_attributes(self.graph, "weight")
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_color="lightblue",
            edge_color="gray",
            node_size=2000,
            font_size=15,
        )
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.show()
        logger.info("Visualized the graph.")

    def graph_statistics(self):
        stats = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "connected_components": nx.number_connected_components(
                self.graph.to_undirected()
            )
            if self.graph.nodes()
            else 0,
        }
        logger.info(f"Graph statistics: {stats}")
        return stats

    def interactive_mode(self):
        while True:
            print("\nNeuro-Symbolic Network Interactive Mode")
            print("1. Add Symbolic Rule")
            print("2. Delete Symbolic Rule")
            print("3. Modify Symbolic Rule")
            print("4. Add Weighted Rule")
            print("5. Reason from Node")
            print("6. Reason with BFS from Node")
            print("7. Find Shortest Path")
            print("8. Reason with Weights from Node")
            print("9. Visualize Graph")
            print("10. Show Graph Statistics")
            print("11. Exit")

            choice = input("Choose an option: ")

            try:
                if choice == "1":
                    rule = input("Enter symbolic rule (format: '1,2 2,3'): ")
                    rule_list = [
                        tuple(map(int, pair.split(","))) for pair in rule.split()
                    ]
                    self.add_symbolic_rule(rule_list)
                elif choice == "2":
                    rule = input("Enter symbolic rule to delete (format: '1,2 2,3'): ")
                    rule_list = [
                        tuple(map(int, pair.split(","))) for pair in rule.split()
                    ]
                    self.delete_symbolic_rule(rule_list)
                elif choice == "3":
                    old_rule = input("Enter old symbolic rule (format: '1,2 2,3'): ")
                    old_rule_list = [
                        tuple(map(int, pair.split(","))) for pair in old_rule.split()
                    ]
                    new_rule = input("Enter new symbolic rule (format: '1,2 2,3'): ")
                    new_rule_list = [
                        tuple(map(int, pair.split(","))) for pair in new_rule.split()
                    ]
                    self.modify_symbolic_rule(old_rule_list, new_rule_list)
                elif choice == "4":
                    rule = input("Enter weighted rule (format: '1,2,1.5 2,3,2.0'): ")
                    rule_list = [
                        tuple(
                            map(
                                lambda x: int(x) if i < 2 else float(x),
                                triple.split(","),
                            )
                        )
                        for i, triple in enumerate(rule.split())
                    ]
                    self.add_weighted_rule(rule_list)
                elif choice == "5":
                    node = int(input("Enter start node: "))
                    print(f"Reasoning path from node {node}:", self.reason(node))
                elif choice == "6":
                    node = int(input("Enter start node: "))
                    print(
                        f"BFS reasoning path from node {node}:", self.reason_bfs(node)
                    )
                elif choice == "7":
                    source = int(input("Enter source node: "))
                    target = int(input("Enter target node: "))
                    print(
                        f"Shortest path from node {source} to node {target}:",
                        self.shortest_path(source, target),
                    )
                elif choice == "8":
                    source = int(input("Enter source node: "))
                    target = int(input("Enter target node: "))
                    print(
                        f"Reasoning path with weights from node {source} to node {target}:",
                        self.reason_with_weights(source, target),
                    )
                elif choice == "9":
                    self.visualize_graph()
                elif choice == "10":
                    print("Graph Statistics:", self.graph_statistics())
                elif choice == "11":
                    break
                else:
                    print("Invalid choice, please try again.")
            except Exception as e:
                logger.error(f"Error during interactive mode: {e}")
                print(f"An error occurred: {e}")
