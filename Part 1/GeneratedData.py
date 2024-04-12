from typing import Callable, Tuple
from functools import partial

import json
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from pathlib import Path

import scipy.stats as stats



class GeneratedData:
    def __init__(
            self, 
            name, 
            func: Callable[[np.ndarray], np.ndarray], 
            x1: Tuple[float, float],
            x2: Tuple[float, float] = None,
            train_split: float = 0.8, 
            noise_level: float = 1.0, 
            outdir='.'
    ):
        self.name = name
        self.outdir = outdir
        self.x1 = x1
        self.x2 = x2

        self.y = func(self.x1, self.x2) + np.random.normal(0, noise_level, size=len(x1))

    def center_data(self):
        self.x = self.x - np.mean(self.x)
        self.y = self.y - np.mean(self.y)

    @staticmethod
    def linear(m: float, b: float, x1: np.ndarray, x2: np.ndarray = None):
        return m * x1 + b

    @staticmethod
    def quadratic(a: float, b: float, c: float, x: np.ndarray):
        return a * x**2 + b*x + c
    
    @staticmethod
    def capped_linear(m: float, b: float, cap: float, x1: np.ndarray):
        return np.minimum(np.maximum(m * x1 + b, cap[0]), cap[1])
    
    @staticmethod
    def capped_linear_2d(m1: float, m2: float, b: float, x1: np.ndarray, x2: np.ndarray):
        res = m1 * x1 + m2 * x2 + b
        res = (res - np.min(res))/(np.max(res) - np.min(res)) * 0.2
        return res
    
    @staticmethod
    def logistic_growth(a: float, b: float, k: float, x: np.ndarray):
        return a / (1 + b * np.exp(-k*x))

    def plot(self, show=False, save=True):
        _fig, ax = plt.subplots()
        ax.scatter(self.x, self.y, color="#fa5a5a")
        if show: plt.show()
        if save: plt.savefig(Path(self.outdir, self.name).with_suffix('.jpg'))

    def save(self):
        with open(Path(self.outdir, self.name+'_x').with_suffix('.json'), 'w') as f:
            json.dump(self.x.tolist(), f)
        with open(Path(self.outdir, self.name+'_y').with_suffix('.json'), 'w') as f:
            json.dump(self.y.tolist(), f)


def main():
    outdir = 'training_data/'

    model_functions = {"habitat_ok": partial(GeneratedData.capped_linear, 0.001, 0.8, (0.05,0.95)),
                       "habitat_ko": partial(GeneratedData.capped_linear, -0.02, 0.9, (0.05,0.95)),
                       "vehicle_ok": partial(GeneratedData.capped_linear, 0.002, 0.8, (0.05,0.95)),
                       "vehicle_ko": partial(GeneratedData.capped_linear, -0.04, 0.9, (0.05,0.95)),
                       "docked_ok": partial(GeneratedData.capped_linear_2d, m1=-2, m2=-1, b=0),  
                        "docked_ko": partial(GeneratedData.capped_linear_2d, m1=-3, m2=-2, b=0)
                      }

    # for name, fn in model_functions.items():
    #     # Habitat models
    #     x1 = np.linspace(0, 1, 1000)
    #     x2 = None
    #     if name.split('_')[0] == "docked":
    #         for h in ["habitat_ko", "habitat_ok"]:
    #             for v in ["vehicle_ko", "vehicle_ok"]:
    #                 with open()
            
    #     data = GeneratedData(
    #             name, 
    #             fn, 
    #             x1 = x1,
    #             x2 = x2, 
    #             noise_level=0.03,
    #             outdir=outdir
    #     )

    #     data.plot()
    #     data.save()
    # }


if __name__ == "__main__":

    main()