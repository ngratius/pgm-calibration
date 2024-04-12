from typing import Callable, Tuple, List, Dict
from functools import partial

import json
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from pathlib import Path

import scipy.stats as stats

from GeneratedData import GeneratedData

class TrialData:
    def __init__(
            self, 
            name, 
            ok_func: Callable[[np.ndarray], np.ndarray], 
            ko_func: Callable[[np.ndarray], np.ndarray], 
            conditional_prob: Dict,
            conditional_data: List['TrialData'],
            timesteps = 2000, 
            n_trials: int = 1000,
            noise_level: float = 0.01, 
            outdir='./generated_trials'
    ):

        self.name = name
        self.outdir = outdir
        
        #Generate time series data
        self.xs = []
        self.ys = []

        t = np.linspace(0,1,timesteps)

        #params
        self.params = {"name":name, "n_trials": n_trials, "status": [], "n_samples": timesteps}

        current_func = None
        for i in range(0, n_trials):
            ko_prob = self.get_ko_prob(conditional_prob, conditional_data, i)

            if np.random.random() >= ko_prob:
                self.params['status'].append(1)
                current_func = ok_func
            else:
                self.params['status'].append(0)
                current_func = ko_func

            #Generate new x and y data
            if name == "docked":
                inputs = [] 
                for data in conditional_data:
                    inputs.append(np.array(data.ys[i]))
            else:
                inputs = [t]
            
            y = current_func(*inputs) 
            
            y += np.random.normal(0, noise_level, size=timesteps)
            
            self.xs.append(inputs)
            self.ys.append(y)
            

    def get_ko_prob(self, conditional_prob, conditional_data, i):
        #Move down the conditional probability table:
        ko_prob = conditional_prob
        for d in conditional_data:
            if d.params["status"][i] == 0:
                ko_prob = ko_prob[f"{d.name}_ko"]
            else: 
                ko_prob = ko_prob[f"{d.name}_ok"]        
        
        return ko_prob

    def plot(self, show=True, save=False):
        fig, axs = plt.subplots(nrows=self.params["n_trials"])
        fig.suptitle(self.params["name"])
        for i, (x, y) in enumerate(zip(self.xs, self.ys)):
            print(x.shape, y.shape)
            axs[i].scatter(x, y, color="#fa5a5a")
            axs[i].set_title("ok" if self.params["status"][i] == 1 else "ko", color = "green" if self.params["status"][i] == 1 else "red")
        # if show: plt.show()
        if save: plt.savefig(Path(self.outdir, self.name).with_suffix('.jpg'))

    def save(self):
        with open(Path(self.outdir, self.name+'_x').with_suffix('.json'), 'w') as f:
            json.dump([[list(xi) for xi in x] for x in self.xs], f)
        with open(Path(self.outdir, self.name+'_y').with_suffix('.json'), 'w') as f:
            json.dump([list(y) for y in self.ys], f)
        with open(Path(self.outdir, self.name+'_params').with_suffix('.json'), 'w') as f:
            json.dump(self.params, f)


if __name__ == "__main__":
    
    with open("./ground_truth/prob_table.json", 'r') as f:
        conditional_probability_table = json.load(f)

    with open("./ground_truth/thetas.json", 'r') as f:
        thetas = json.load(f)

    H_ok = partial(getattr(GeneratedData, thetas["habitat_ok"]["method"]),*thetas['habitat_ok']["params"])
    H_ko = partial(getattr(GeneratedData, thetas["habitat_ko"]["method"]),*thetas['habitat_ko']["params"])
    habitat_conditional_prob = conditional_probability_table["habitat_ko"]
    habitat_data = TrialData("habitat", H_ok, H_ko, habitat_conditional_prob, [])
    # habitat_data.plot()
    habitat_data.save()

    V_ok = partial(getattr(GeneratedData, thetas["vehicle_ok"]["method"]),*thetas['vehicle_ok']["params"])
    V_ko = partial(getattr(GeneratedData, thetas["vehicle_ko"]["method"]),*thetas['vehicle_ko']["params"])
    vehicle_conditional_prob = conditional_probability_table["vehicle_ko"]
    vehicle_data = TrialData("vehicle", V_ok, V_ko, vehicle_conditional_prob, [habitat_data])
    # vehicle_data.plot()
    vehicle_data.save()

    D_ok = partial(getattr(GeneratedData, thetas["docked_ok"]["method"]),*thetas['docked_ok']["params"])
    D_ko = partial(getattr(GeneratedData, thetas["docked_ko"]["method"]),*thetas['docked_ko']["params"])
    docked_conditional_prob = conditional_probability_table["docked_ko"]
    docked_data = TrialData("docked", D_ok, D_ko, docked_conditional_prob, [habitat_data, vehicle_data])
    # docked_data.plot()
    docked_data.save()

    plt.show()
