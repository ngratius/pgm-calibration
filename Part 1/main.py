from LinearRegression import CappedLinearRegression
import json
from pathlib import Path
from ParameterModel import ParameterModel
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('ggplot')

class PGM:
    def __init__(self):
        self.data_dir = 'training_data/'
        self.model_dir = 'models/'
        self.model_names = ["habitat_ok", "habitat_ko", "vehicle_ok", "vehicle_ko", "docked_ok", "docked_ko"]

        self.theta_vehicle_new, self.theta_habitat_new, self.theta_docked_new = (None, None, None)
        self.trained_models = {}



    def train(self):
        for model_name in self.model_names:
            data_x = json.load(open(Path(self.data_dir, model_name + "_x").with_suffix('.json')))
            data_y = json.load(open(Path(self.data_dir, model_name + "_y").with_suffix('.json')))

            model = CappedLinearRegression()
            model.load_data(X=data_x, y=data_y)

            model.train()
            print(f"{model.params() = }")

            model.save(model_name, self.model_dir)

    def load_all_models(self):
        for model_path in glob.glob(f"./{self.model_dir}*.pkl"):
            model_name = model_path.split(os.sep)[-1].split('.')[0]
            model = CappedLinearRegression()
            model.load(model_name, self.model_dir)
            self.trained_models[model_name] = model


    def update(self):
        print(f"{self.trained_models = }")

        learned_params = {
            'theta_vehicle_ko': self.trained_models['vehicle_ko'].params(),
            'theta_vehicle_ok': self.trained_models['vehicle_ok'].params(),
            'theta_habitat_ko': self.trained_models['habitat_ko'].params(),
            'theta_habitat_ok': self.trained_models['habitat_ok'].params(),
            'theta_docked_ko': self.trained_models['docked_ko'].params(),
            'theta_docked_ok': self.trained_models['docked_ok'].params()
        }

        parameter_model = ParameterModel(learned_params)

        # Assumed input from diagnosis module:
        diagnosis_result = 'theta_habitat_ko'

        # Parameters of individual models:
        self.theta_vehicle_new, self.theta_habitat_new, self.theta_docked_new = parameter_model.infer(
            diagnosis_result)

        # print(f"{self.theta_vehicle_new =}, {self.theta_habitat_new =}, {self.theta_docked_new =}")

    def simulate(self, time_horizon, n_samples):
        vehicle_new_model = CappedLinearRegression()
        vehicle_new_model.set_params(self.theta_vehicle_new)
        habitat_new_model = CappedLinearRegression()
        habitat_new_model.set_params(self.theta_habitat_new)
        docked_new_model = CappedLinearRegression()
        docked_new_model.set_params(self.theta_docked_new)

        X = np.linspace(0, time_horizon, n_samples).reshape(-1, 1)
        y_vehicle = vehicle_new_model.predict(X)
        y_habitat = habitat_new_model.predict(X)
        y_docked = docked_new_model.predict(X)

        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(212)

        fig.suptitle('Model outputs', fontweight='bold')

        ax1.scatter(X, y_vehicle, s=5)
        ax2.scatter(X, y_habitat, s=5)
        ax3.scatter(X, y_docked, s=5)

        ax1.title.set_text('Vehicle CO2 removal capacity')
        ax2.title.set_text('Station CO2 removal capacity')
        ax3.title.set_text('CO2 in habitat after docking')

        ax1.set(xlabel='Time', ylabel='Yield [%]')
        ax2.set(xlabel='Time', ylabel='Yield [%]')
        ax3.set(xlabel='Time', ylabel='Concentration [%]')

        plt.tight_layout()
        plt.show()



def main():
    time_horizon = 50
    n_samples = 100

    simulator = PGM()
    simulator.train()
    simulator.load_all_models()
    simulator.update()
    simulator.simulate(time_horizon, n_samples)

if __name__ == '__main__':
    main()






