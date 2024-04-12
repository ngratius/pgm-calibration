# Source: https://pgmpy.org/detailed_notebooks/2.%20Bayesian%20Networks.html

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from typing import Dict, Tuple

from typing import Callable, Tuple, List, Dict
from GeneratedData import GeneratedData
from functools import partial
import numpy as np
import json

class PGModel():
    def __init__(
        self,
        name,
        func_0: Callable[[np.ndarray], np.ndarray],
        func_1: Callable[[np.ndarray], np.ndarray],
        statuses: List[int]
    ):
        self.name = name
        self.func_0 = func_0
        self.func_1 = func_1
        self.statuses = statuses

    def predict(self, x, status):
        func = None
        func = self.func_1 if status else self.func_0

        y = func(*x)
        return y

    def predict_all(self, xs):
        ys = []
        for x, status in zip(xs, self.statuses):
            y = self.predict(x, status)
            ys.append(y)

        return ys