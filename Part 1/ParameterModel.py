# Source: https://pgmpy.org/detailed_notebooks/2.%20Bayesian%20Networks.html

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from typing import Dict, Tuple
from typing import Callable, Tuple, List, Dict
from GeneratedData import GeneratedData
from functools import partial
import numpy as np


class ParameterModel:
    def __init__(
            self, 
            learned_params: Dict[str, Callable[[np.ndarray], np.ndarray]],
            conditional_probability_table: Dict
        ):
        self.learned_params = learned_params
        # 1. PGM DEFINITION:
        # Define edges

        self.parameter_connections = [('theta_habitat', 'theta_vehicle'), (
            'theta_habitat', 'theta_docked'), ('theta_vehicle', 'theta_docked')]
        self.model = BayesianNetwork(self.parameter_connections)

        # Define probabilities (from expert knowledge)
        
        # probability habitat ko
        self.p1 = conditional_probability_table["habitat_ko"]
        
        # probability vehicle ko given habitat ko
        self.p2 = conditional_probability_table["vehicle_ko"]["habitat_ko"]
        
        # probability vehicle ko given habitat ok
        self.p3 = conditional_probability_table["vehicle_ko"]["habitat_ok"]
        
        # probability docked ko given habitat ko, vehicle ko
        self.p4 = conditional_probability_table["docked_ko"]["habitat_ko"]["vehicle_ko"]
        
        # probability docked ko given habitat ko, vehicle ok
        self.p5 = conditional_probability_table["docked_ko"]["habitat_ko"]["vehicle_ok"]
        
        # probability docked ko given habitat ok, vehicle ko
        self.p6 = conditional_probability_table["docked_ko"]["habitat_ok"]["vehicle_ko"]
        
        # probability docked ko given habitat ok, vehicle ok
        self.p7 = conditional_probability_table["docked_ko"]["habitat_ok"]["vehicle_ok"]

        # Define CPDs (columns are evidences and rows are variable states)
        cpd_theta_habitat = TabularCPD(variable='theta_habitat', variable_card=2,
                                       values=[[self.p1],
                                               [1 - self.p1]],
                                       state_names={'theta_habitat': ['theta_habitat_ko', 'theta_habitat_ok']})

        cpd_theta_vehicle = TabularCPD(variable='theta_vehicle', variable_card=2,
                                       values=[[self.p2, self.p3],
                                               [1 - self.p2, 1 - self.p3]],
                                       evidence=['theta_habitat'], evidence_card=[2],
                                       state_names={'theta_vehicle': ['theta_vehicle_ko', 'theta_vehicle_ok'],
                                                    'theta_habitat': ['theta_habitat_ko', 'theta_habitat_ok']})

        cpd_theta_docked = TabularCPD(variable='theta_docked', variable_card=2,
                                        values=[[self.p4, self.p5, self.p6, self.p7],
                                                [1 - self.p4, 1 - self.p5, 1 - self.p6, 1 - self.p7]],
                                        evidence=['theta_vehicle', 'theta_habitat'], evidence_card=[2, 2],
                                        state_names={'theta_docked': ['theta_docked_ko', 'theta_docked_ok'],
                                                     'theta_vehicle': ['theta_vehicle_ko', 'theta_vehicle_ok'],
                                                     'theta_habitat': ['theta_habitat_ko', 'theta_habitat_ok']})

        # Associate CPDs with the network
        self.model.add_cpds(cpd_theta_vehicle,
                            cpd_theta_habitat, cpd_theta_docked)

        # Check model structure and param. sum to 1
        self.model.check_model()

    def infer(self, diagnosis_eclss_2):
        # 2.INFERENCE
        infer = VariableElimination(self.model)

        # Most likely state
        state_theta_vehicle = infer.map_query(['theta_vehicle'], evidence={
                                              'theta_habitat': diagnosis_eclss_2})['theta_vehicle']
        state_theta_docked = infer.map_query(['theta_docked'], evidence={
                                               'theta_habitat': diagnosis_eclss_2})['theta_docked']

        # Dictionary mapping states to parameter values:

        # theta_habitat = self.learned_params['theta_habitat_ko']
        theta_habitat = self.learned_params['theta_habitat_ok' if diagnosis_eclss_2 else 'theta_habitat_ko']
        theta_vehicle, theta_docked = self.learned_params[state_theta_vehicle], self.learned_params[state_theta_docked]

        return theta_vehicle, theta_habitat, theta_docked