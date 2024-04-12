from DAGGenerate import DAGs_generate, plot_DAG
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.sampling import GibbsSampling, BayesianModelSampling
from pgmpy.factors.discrete import State

from typing import Dict, Tuple
from typing import Callable, Tuple, List, Dict
from functools import partial
import numpy as np
import scipy

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

class Node:
    '''
        Node class storing conditional probability table, node information and parents.
    '''
    def __init__(self, name, cardinality):
        self.cardinality = cardinality
        self.name = name
        self.values = np.arange(0, cardinality)
        self.parents = []
        
        self.probability_table = None
        np.random.seed(1)

    def add_parent(self, p):
        self.parents.append(p)

    def fill_probability_table(self):
        '''
            Fills probability table using dirichlet distribution.
        '''
        parents_cardinality = [p.cardinality for p in self.parents]
        multi_d_probability_table = np.random.dirichlet(np.ones(self.cardinality) / self.cardinality, 
                                                     size=(*parents_cardinality,)) 
        
        self.probability_table = np.reshape(multi_d_probability_table, (np.prod(parents_cardinality, initial=1, dtype=int), self.cardinality)).T

    def __repr__(self):
        if len(self.parents) > 0:
            return f"{self.name}: parents={[p.name for p in self.parents]}"
        else:
            return f"{self.name}"
    

class ParameterModel:
    '''
        Class representing the PGM.
    '''
    def __init__(self, num_nodes, node_cardinality):
        self.node_cardinality = node_cardinality
        self.num_nodes = num_nodes

        self.position_by_prefix = {}

        self.habitat_edges, self.habitat_nodes, self.habitat_node_by_id = self._generate_sub_graph(prefix="h", n=num_nodes, alpha=1, beta=1)
        self.vehicle_edges, self.vehicle_nodes, self.vehicle_node_by_id = self._generate_sub_graph(prefix="v", n=num_nodes, alpha=1, beta=1)
        self.combined_edges, self.combined_nodes, self.combined_node_by_id = self._generate_sub_graph(prefix="c", n=num_nodes, alpha=1, beta=1)

        
        self.all_edges, self.all_nodes, self.all_node_by_id = self._connect_graphs()

        self._update_graph_positions()


        for node in self.all_nodes:
            node.fill_probability_table()

    def _update_graph_positions(self):
        '''
            Updates node positions to improve graph visualization.
        '''

        habitat_names, habitat_positions = zip(*self.position_by_prefix['h'].items())
        habitat_positions = np.array(list(habitat_positions))
        vehicle_names, vehicle_positions = zip(*self.position_by_prefix['v'].items())
        vehicle_positions = np.array(list(vehicle_positions))
        combined_names, combined_positions = zip(*self.position_by_prefix['c'].items())
        combined_positions = np.array(list(combined_positions))

        #Rotate
        combined_orig = combined_positions.copy()
        combined_positions[:, 0] = combined_orig[:, 1]
        combined_positions[:, 1] = combined_orig[:, 0]*2

        vehcle_orig = vehicle_positions.copy()
        vehicle_positions[:, 0] = vehcle_orig[:, 1]
        vehicle_positions[:, 1] = vehcle_orig[:, 0]*2
        vehicle_positions[:, 1] = np.max(vehicle_positions[:, 1]) - vehicle_positions[:, 1]#flip

        habitat_orig = habitat_positions.copy()
        habitat_positions[:, 0] = habitat_orig[:, 1]
        habitat_positions[:, 1] = habitat_orig[:, 0]*2
        habitat_positions[:, 1] = np.max(habitat_positions[:, 1]) - habitat_positions[:, 1]#flip


        #Position Relative
        habitat_x_max, habitat_y_max = np.max(habitat_positions, axis=0)
  
        const = 10
        vehicle_positions[:,0] += habitat_x_max + const

        vehicle_x_max, vehicle_y_max = np.max(vehicle_positions, axis=0)

        combined_positions[:,1] -= 0.7*max(habitat_y_max, vehicle_y_max) + const
        combined_positions[:,0] += vehicle_x_max * 0.5 - 0.5*(np.max(combined_positions[:,0]))

        #Offset c_CO2 (opional)
        # c_CO2_idx = combined_names.index("c_CO2")
        # print(c_CO2_idx)
        # combined_positions[c_CO2_idx, 1] += const
        
        #Convert numpy positions back to dictionary
        self.position_by_prefix['a'] = {}
        for name, pos in zip(habitat_names, habitat_positions):
            self.position_by_prefix['a'][name] = tuple(pos)
        for name, pos in zip(vehicle_names, vehicle_positions):
            self.position_by_prefix['a'][name] = tuple(pos)
        for name, pos in zip(combined_names, combined_positions):
            self.position_by_prefix['a'][name] = tuple(pos)
        
    def _connect_graphs(self):
        '''
            Connects habitat and vehicle and combined graphs to form a single graph.
        '''

        all_edges = self.habitat_edges + self.vehicle_edges + self.combined_edges
        all_nodes = self.habitat_nodes + self.vehicle_nodes + self.combined_nodes
        all_node_by_id = self.combined_node_by_id | self.vehicle_node_by_id | self.habitat_node_by_id

        #connect CO2 habitat and vehicle to combined
        h_CO2 = all_node_by_id['h_CO2']
        v_CO2 = all_node_by_id['v_CO2']
        c_CO2 = all_node_by_id['c_CO2']

        all_edges.append(("h_CO2", "c_CO2"))
        c_CO2.parents.append(h_CO2)
        all_edges.append(("v_CO2", "c_CO2"))
        c_CO2.parents.append(v_CO2)

        return all_edges, all_nodes, all_node_by_id

    def _nodes_from_edges(self, edges):
        '''
            Given a list of edges, returns a list of nodes.
        '''
        nodes = []
        seen_node_names = set()
        for (src, dest) in edges:
            if src not in seen_node_names:
                nodes.append(Node(name=src, cardinality=self.node_cardinality))
                seen_node_names.add(src)
        for (src, dest) in edges:
            if dest not in seen_node_names:
                nodes.append(Node(name=dest, cardinality=self.node_cardinality))
                seen_node_names.add(dest)
        return nodes

    """
    Generate a DAG given some parameters.
    Source: https://github.com/Livioni/DAG_Generator
    """
    def _node_mapping(self, label, prefix):
        '''
            Creates descriptive node names.
        '''
        if label == 'Start':
            return f"{prefix}_0"
        elif label == 'Exit':
            return f"{prefix}_CO2"
        else:
            return f"{prefix}_{label}"
        
    def _generate_sub_graph(self, prefix, n=10, alpha=1, beta=1):
        '''
            Creates graph in the form of node objects.
            args: 
                prefix (str): prefix for node names
                n (int): number of nodes
                alpha (float): alpha parameter for DAG generator
                beta (float): beta parameter for DAG generator
        '''
        edges = self._generate_dag(n=n, alpha=alpha, beta=beta, prefix=prefix)
        nodes = self._nodes_from_edges(edges)
        node_by_id = {node.name:node for node in nodes}
        for (source, destination) in edges:
            node_by_id[destination].add_parent(node_by_id[source])
        
        return edges, nodes, node_by_id
        
    def _generate_dag(self, prefix="", n=10, alpha=1, beta=1):
        '''
            Generates a DAG as a list of edges using the DAG generator.
            args:
                prefix (str): prefix for node names
                n (int): number of nodes
                alpha (float): alpha parameter for DAG generator
                beta (float): beta parameter for DAG generator
            returns:
                modified_edges (list): list of edges
        '''
        # Randomly generate a DAG using external library
        edges, in_degree, out_degree, position = DAGs_generate(n=n, max_out=1, alpha=alpha, beta=beta, seed_prefix=prefix)
        positions = {str(k): v for k,v in position.items()}
        # Convert the edges to have names that we expect them to
        edges = [(str(edge[0]), str(edge[1])) for edge in edges]
        node_mapping = {}
        for src, dest in edges:
            node_mapping[src] = self._node_mapping(label=src, prefix=prefix)
            node_mapping[dest] = self._node_mapping(label=dest, prefix=prefix)
        modified_positions = {}
        for k, v in positions.items():
            modified_positions[node_mapping[k]] = v
        self.position_by_prefix[prefix] = modified_positions
        # print(f"{modified_positions =}")
        modified_edges = [(node_mapping[src], node_mapping[dest]) for (src, dest) in edges]
        return modified_edges

    
    def plot(self):
        '''
            Plots the final generated graph.
        '''
        habitat_names = list(self.habitat_node_by_id.keys())
        vehicle_names = list(self.vehicle_node_by_id.keys())
        combined_names = list(self.combined_node_by_id.keys())

        fname='results/graph_structure'

        #COlORS
        color_map = {"habitat": "#f55953", "vehicle": "#87d466", "combined": "#9dccfa", "observed": "#8a8a8a", "predicted": "#8c78f0"}
        nodelist = habitat_names + vehicle_names + combined_names
        node_color = [color_map["habitat"] for i in range(len(habitat_names))] + [color_map["vehicle"] for i in range(len(vehicle_names))] + [color_map["combined"] for i in range(len(combined_names))]
        
        #C
        obs_node = nodelist.index("h_0")
        node_color[obs_node] = color_map["observed"]
        obs_node = nodelist.index("c_CO2")
        node_color[obs_node] = color_map["predicted"]

        #PLOT
        plot_DAG(edges=self.all_edges, postion=self.position_by_prefix['a'], fname=fname, nodelist=nodelist, node_color=node_color, size=self.num_nodes)


        #LEGEND
        legend_elements = [Line2D([0], [0], marker='*', color='w', label="Diagnosis Observation",
                          markerfacecolor=color_map["observed"], markersize=15),
                          Line2D([0], [0], marker='*', color='w', label="Final Combined CO2 Prediction",
                          markerfacecolor=color_map["predicted"], markersize=15),
                          Line2D([0], [0], marker='o', color='w', label="Habitat Variable",
                          markerfacecolor=color_map["habitat"], markersize=15), 
                          Line2D([0], [0], marker='o', color='w', label="Vehicle Variable",
                          markerfacecolor=color_map["vehicle"], markersize=15),
                          Line2D([0], [0], marker='o', color='w', label="Combined Variable",
                          markerfacecolor=color_map["combined"], markersize=15),
                          ]

        plt.legend(handles=legend_elements, prop={'size': 15})


        #SAVE
        plt.savefig(f"{fname}.png", format="png")
        return plt.clf

    def print_full(self, cpd, f):
        '''
            Prints the full CPD table to a file.
        '''
        backup = TabularCPD._truncate_strtable
        TabularCPD._truncate_strtable = lambda self, x: x
        # print(cpd)
        f.write(str(cpd))
        f.write("\n\n")
        TabularCPD._truncate_strtable = backup
    
    def construct_pgmpy(self):
        '''
            Constructs the pgmpy model from the generated graph and CPD tables.
        '''
        self.model = BayesianNetwork(self.all_edges)

        cpds = []

        for node in self.all_nodes:
            evidence = [p.name for p in node.parents] if len(node.parents) > 0 else None
            evidence_card = [p.cardinality for p in node.parents] if len(node.parents) > 0 else None

            cpd = TabularCPD(
                variable=node.name, 
                variable_card=node.cardinality,
                values=node.probability_table,
                evidence=evidence, 
                evidence_card=evidence_card
            )
            # print(cpd)
            cpds.append(cpd)


        # Print conditional probability tables
        with open('results/cpd.txt', 'w') as f:
            for cpd in cpds:
                self.print_full(cpd, f)

        # Associate CPDs with the network
        self.model.add_cpds(*cpds)
        
        # Check model structure and param. sum to 1
        self.model.check_model()

    def exact_infer(self, evidence_var, evidence_state, show_progress=True):
        '''
            Runs exact inference on the model.
        '''
        # Run inference
        infer = VariableElimination(self.model)

        # Most likely state for each node
        states = {}
        states[evidence_var] = evidence_state
        for node in self.all_nodes:
            if node.name not in states:
                node_state = infer.map_query(variables=[node.name], evidence={evidence_var: evidence_state}, show_progress=show_progress)[node.name]
                states[node.name] = node_state

        return states
    
    def gibbs_infer(self, evidence_var, evidence_state, show_progress=True, n_samples=1000):
        '''
            Runs Gibbs sampling on the model using giibs sampling.
            args:
                evidence_var (str): The variable to condition on.
                evidence_state (int): The state of the variable to condition on.
                show_progress (bool): Whether to show progress of the inference.
                n_samples (int): The number of samples to generate.
            returns:
                gen (dict): A dictionary of the most likely state for each node.
        '''
        infer = GibbsSampling(self.model)
        gen = infer.generate_sample(size=n_samples)

        # gibbs_chain = GibbsSampling(self.model)
        # gibbs_chain.sample(size=n_samples)

        print(f"Before filtering: {len(gen)} samples")
        filtered_samples = [sample for sample in gen if sample[evidence_var] == evidence_state]
        print(f"After filtering: {len(gen)} samples")
        # print(infer)

        # for sample in gen:
        #     print(sample)
        # Most likely state for each node
        # states = {}
        # states[evidence_var] = evidence_state
        # for node in self.all_nodes:
        #     if node.name not in states:
        #         node_state = infer.query(variables=[node.name], evidence={evidence_var: evidence_state}, show_progress=show_progress)[node.name]
        #         states[node.name] = node_state

        return gen

    def likelihood_weighted_sample(self, evidence_var, evidence_state, show_progress=True, n_samples=1000):
        '''
            Runs likelihood weighted sampling on the model.
            args:
                evidence_var (str): The variable to condition on.
                evidence_state (int): The state of the variable to condition on.
                show_progress (bool): Whether to show progress of the inference.
                n_samples (int): The number of samples to generate.
            returns:
                gen (dict): A dictionary of the most likely state for each node.
        '''
        np.random.seed(None)
        #Run inference
        infer = BayesianModelSampling(self.model)
        evidence = [State(evidence_var, evidence_state)]
        gen = infer.likelihood_weighted_sample(evidence=evidence, size=n_samples, show_progress=show_progress)
        
        return gen


    def rejection_sample(self, evidence_var, evidence_state, show_progress=True, n_samples=1000):
        '''
            Runs rejection sampling on the model.
            args:
                evidence_var (str): The variable to condition on.
                evidence_state (int): The state of the variable to condition on.
                show_progress (bool): Whether to show progress of the inference.
                n_samples (int): The number of samples to generate.
            returns:
                gen (dict): A dictionary of the most likely state for each node.
        '''
        infer = BayesianModelSampling(self.model)
        evidence = [State(evidence_var, evidence_state)]
        gen = infer.rejection_sample(evidence=evidence, size=n_samples, show_progress=show_progress)
        return gen


    def forward_sample(self, evidence_var, evidence_state, show_progress=True, n_samples=1000):
        '''
            Runs forward sampling on the model.
            args:
                evidence_var (str): The variable to condition on.
                evidence_state (int): The state of the variable to condition on.
                show_progress (bool): Whether to show progress of the inference.
                n_samples (int): The number of samples to generate.
            returns:
                gen (dict): A dictionary of the most likely state for each node.
        '''
        modified_model = self.model.copy()

        #Remove CPD for evidence var
        h_0_cpd = [cpd for cpd in modified_model.cpds if cpd.variable == 'h_0'][0]
        modified_model.remove_cpds(h_0_cpd)
        values = np.zeros((h_0_cpd.variable_card,1))
        values[evidence_state] = 1
        new_h_0_cpd = TabularCPD(
                variable=h_0_cpd.variable, 
                variable_card=h_0_cpd.variable_card,
                values=values

        )
        modified_model.add_cpds(new_h_0_cpd)

        # Run inference on modified model
        infer = BayesianModelSampling(modified_model)
        gen = infer.forward_sample(size=n_samples, show_progress=show_progress)
        return gen


    def sample_inference(self, gen, nodes):
        '''
            Given a set of samples, returns the most likely state for each node.
            args:
                gen (dict): A dictionary of the most likely state for each node.
                nodes (list): A list of nodes to get predictions for.
            returns:
                all_predictions (dict): A dictionary of the most likely state for each node.
        '''
        all_predictions = {}
        for node in nodes:
            unique, counts = np.unique(gen[node],return_counts=True)
            max_idx = np.argmax(counts)
            prediction = unique[max_idx]

            all_predictions[node] = prediction
        
        return all_predictions
    
    def approx_vs_exact(self, approx_predictions, exact_predictions):
        '''
            Given a set of approximate predictions and exact predictions, returns the accuracy of the approximate predictions.
            args:
                approx_predictions (dict): A dictionary of the most likely state for each node.
                exact_predictions (dict): A dictionary of the most likely state for each node.
            returns:    
                correct / total (float): The accuracy of the approximate predictions.
        '''
        correct = 0
        total = 0
        for k, v in exact_predictions.items():
            approx_pred = approx_predictions[k]
            correct += approx_pred == v
            total += 1

        return correct / total
    
if __name__ == "__main__":
    model = ParameterModel(num_nodes=15, node_cardinality=3)
    model.plot()
