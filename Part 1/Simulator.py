from typing import Callable, Tuple, List, Dict
from RandomBaseline import RandomBaseline
from MarginalBaseline import MarginalBaseline
from ParameterModel import ParameterModel
from PGModel import PGModel
from GeneratedData import GeneratedData
from functools import partial
import numpy as np
import json

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class Simulator:
    def __init__(self):
        pass
    
    def generate_data(self):
        assert "Incomplete function"
    
    def load_data(self):
        with open("./ground_truth/thetas.json", 'r') as f:
            self.thetas = json.load(f)

        with open("./ground_truth/prob_table.json", 'r') as f:
            self.conditional_probability_table = json.load(f)

        self.H_ok = partial(getattr(GeneratedData, self.thetas["habitat_ok"]["method"]),*self.thetas['habitat_ok']["params"])
        self.H_ko = partial(getattr(GeneratedData, self.thetas["habitat_ko"]["method"]),*self.thetas['habitat_ko']["params"])
        with open("./generated_trials/habitat_x.json") as f:
            self.habitat_xs = [np.array(x) for x in json.load(f)]
        with open("./generated_trials/habitat_y.json") as f:
            self.habitat_y = [np.array(x) for x in json.load(f)]
        with open("./generated_trials/habitat_params.json") as f:
            self.habitat_params = json.load(f)
        self.habitat_conditional_prob = self.conditional_probability_table["habitat_ko"]
        
        self.V_ok = partial(getattr(GeneratedData, self.thetas["vehicle_ok"]["method"]),*self.thetas['vehicle_ok']["params"])
        self.V_ko = partial(getattr(GeneratedData, self.thetas["vehicle_ko"]["method"]),*self.thetas['vehicle_ko']["params"])
        with open("./generated_trials/vehicle_x.json") as f:
            self.vehicle_xs = [np.array(x) for x in json.load(f)]
        with open("./generated_trials/vehicle_y.json") as f:
            self.vehicle_y = [np.array(x) for x in json.load(f)]
        self.vehicle_conditional_prob = self.conditional_probability_table["vehicle_ko"]

        self.D_ok = partial(getattr(GeneratedData, self.thetas["docked_ok"]["method"]),*self.thetas['docked_ok']["params"])
        self.D_ko = partial(getattr(GeneratedData, self.thetas["docked_ko"]["method"]),*self.thetas['docked_ko']["params"])
        with open("./generated_trials/docked_x.json") as f:
            self.docked_xs = [np.array(x) for x in json.load(f)]
        with open("./generated_trials/docked_y.json") as f:
            self.docked_y = [np.array(x) for x in json.load(f)]
        self.docked_conditional_prob = self.conditional_probability_table["docked_ko"]

    def evaluate(self, predictions, ground_truth):
        mses = []
        for Y_pred, Y_true in zip(predictions, ground_truth):
            mse = mean_squared_error(Y_true,Y_pred)
            mses.append(mse)

        return np.mean(mses)

    def run_baselines(self):
        for CLASS_NAME in [RandomBaseline, MarginalBaseline]:
            print()
            print(CLASS_NAME.__name__)
            habitat_model = CLASS_NAME("habitat", self.H_ok, self.H_ko)
            vehicle_model = CLASS_NAME("vehicle", self.V_ok, self.V_ko)
            docked_model = CLASS_NAME("docked", self.D_ok, self.D_ko)

            habitat_pred = habitat_model.predict_all(self.habitat_xs)
            vehicle_pred = vehicle_model.predict_all(self.vehicle_xs)
            docked_pred = docked_model.predict_all(zip(habitat_pred, vehicle_pred))

            mse_habitat = self.evaluate(habitat_pred, self.habitat_y)
            mse_vehicle = self.evaluate(vehicle_pred, self.vehicle_y)
            mse_docked = self.evaluate(docked_pred, self.docked_y)

            print(f"{mse_habitat=:6f}, {mse_vehicle=:6f}, {mse_docked=:6f}")

    def run_pgm(self):
        model = ParameterModel(
            learned_params={
                'theta_vehicle_ko': self.V_ko,
                'theta_vehicle_ok': self.V_ok,
                'theta_habitat_ko': self.H_ko,
                'theta_habitat_ok': self.H_ok,
                'theta_docked_ko': self.D_ko,
                'theta_docked_ok': self.D_ok
            },
            conditional_probability_table=self.conditional_probability_table
        )
        V0, H0, D0 = model.infer(0)
        V1, H1, D1 = model.infer(1)

        habitat_model = PGModel("habitat", H0, H1, self.habitat_params["status"])
        vehicle_model = PGModel("vehicle", V0, V1, self.habitat_params["status"])
        docked_model = PGModel("docked", D0, D1, self.habitat_params["status"])

        self.pgm_habitat_pred = habitat_model.predict_all(self.habitat_xs)
        self.pgm_vehicle_pred = vehicle_model.predict_all(self.vehicle_xs)

        self.pgm_docked_pred = docked_model.predict_all(zip(self.pgm_habitat_pred, self.pgm_vehicle_pred))

        mse_habitat = self.evaluate(self.pgm_habitat_pred, self.habitat_y)
        mse_vehicle = self.evaluate(self.pgm_vehicle_pred, self.vehicle_y)
        mse_docked = self.evaluate(self.pgm_docked_pred, self.docked_y)

        print(f"{mse_habitat=:6f}, {mse_vehicle=:6f}, {mse_docked=:6f}")

    def plot(self, i):
        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(212)

        fig.suptitle('Model outputs', fontweight='bold')

        ax1.scatter(self.vehicle_xs[i][0], self.pgm_vehicle_pred[i], s=5)
        ax2.scatter(self.habitat_xs[i][0], self.pgm_habitat_pred[i], s=5)
        ax3.scatter(self.habitat_xs[i][0], self.pgm_docked_pred[i], s=5)

        ax1.title.set_text('Vehicle CO2 removal capacity')
        ax2.title.set_text('Station CO2 removal capacity')
        ax3.title.set_text('CO2 in habitat after docking')

        ax1.set(xlabel='Time', ylabel='Yield [%]')
        ax2.set(xlabel='Time', ylabel='Yield [%]')
        ax3.set(xlabel='Time', ylabel='Concentration [%]')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    simulator = Simulator()
    simulator.load_data()
    simulator.run_baselines()
    simulator.run_pgm()
    
    simulator.plot(i=4)
    