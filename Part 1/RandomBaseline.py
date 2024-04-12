from typing import Callable, Tuple, List, Dict
from GeneratedData import GeneratedData
from functools import partial
import numpy as np
import json



class RandomBaseline():
    def __init__(
        self,
        name, 
        ok_func: Callable[[np.ndarray], np.ndarray], 
        ko_func: Callable[[np.ndarray], np.ndarray]
    ):
        self.name = name

        self.ok_func = ok_func
        self.ko_func = ko_func

    def predict(self, x):
        func = None
        if np.random.random() > 0.5:
            func = self.ok_func
        else:
            func = self.ko_func

        y = func(*x)
        return y

    def predict_all(self, xs):
        ys = []
        for x in xs:
            y = self.predict(x)
            ys.append(y)

        return ys
