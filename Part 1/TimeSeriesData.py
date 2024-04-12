from typing import Callable, Tuple, List, Dict
from functools import partial

import json
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from pathlib import Path

import scipy.stats as stats

class TimeSeriesData:
    def __init__(
            self, 
            name, 
            ok_func: Callable[[np.ndarray], np.ndarray], 
            ko_func: Callable[[np.ndarray], np.ndarray], 
            conditional_prob: Dict,
            conditional_data: List['TimeSeriesData'],
            n_samples = 2000, 
            x_range: Tuple[float, float] = (0,100), 
            n_splits: int = 40,
            noise_level: float = 1.0, 
            outdir='.'
    ):
        if ok_func(0) != 0 or ko_func(0) != 0:
            raise Exception("Functions must have intercept of 0")

        self.name = name
        self.outdir = outdir
        
        #Generate time series data
        self.x = []
        self.y = []
        #Starting points for x and y
        prev_y = 0
        prev_x = 0
        #params
        self.params = {"name":name, "split_interval": n_splits, "status": [], "n_samples": n_samples}

        split_interval = n_samples // n_splits

        current_func = ok_func
        for i in range(0, n_splits):
            ko_prob = self.get_ko_prob(conditional_prob, conditional_data, i)

            if np.random.random() >= ko_prob:
                self.params['status'].append(1)
                current_func = ok_func
            else:
                self.params['status'].append(0)
                current_func = ko_func

            #Generate new x and y data
            x = np.linspace(0, x_range[1] / n_splits, split_interval)
            y = current_func(x) 
            
            prev_y = y[-1] + prev_y
            prev_x = x[-1] + prev_x

            x += prev_x
            y += prev_y + np.random.normal(0, noise_level, size=split_interval)

            self.x += list(x)
            self.y += list(y)
            

    def get_ko_prob(self, conditional_prob, conditional_data, i):
        #Move down the conditional probability table:
        ko_prob = conditional_prob
        for d in conditional_data:
            if d.params["status"][i] == 0:
                ko_prob = ko_prob[f"{d.name}_ko"]
            else: 
                ko_prob = ko_prob[f"{d.name}_ok"]        

        #If data is no conditioned on anything
        if len(conditional_data) == 0:
            ko_prob = conditional_prob  
        
        return ko_prob

    def plot(self, show=True, save=False):
        _fig, ax = plt.subplots()
        ax.scatter(self.x, self.y, color="#fa5a5a")
        # if show: plt.show()
        if save: plt.savefig(Path(self.outdir, self.name).with_suffix('.jpg'))

    def save(self):
        with open(Path(self.outdir, self.name+'_x').with_suffix('.json'), 'w') as f:
            json.dump(self.x.tolist(), f)
        with open(Path(self.outdir, self.name+'_y').with_suffix('.json'), 'w') as f:
            json.dump(self.y.tolist(), f)
        with open(Path(self.outdir, self.name+'_params').with_suffix('.json'), 'w') as f:
            json.dump(self.params, f)


if __name__ == "__main__":
    
    with open("./data/prob_table.json", 'r') as f:
        conditional_probability_table = json.load(f)

    habitat_conditional_prob = conditional_probability_table["habitat_ko"]
    habitat_data = TimeSeriesData("habitat", lambda x : 0.2*x, lambda x : -1*x, habitat_conditional_prob, [])
    habitat_data.plot()

    vehicle_conditional_prob = conditional_probability_table["vehicle_ko"]
    vehicle_data = TimeSeriesData("vehicle", lambda x : 0.2*x, lambda x : -1*x, vehicle_conditional_prob, [habitat_data])
    vehicle_data.plot()

    combined_conditional_prob = conditional_probability_table["combined_ko"]
    combined_data = TimeSeriesData("combined", lambda x : 0.2*x, lambda x : -1*x, combined_conditional_prob, [habitat_data, vehicle_data])
    combined_data.plot()
