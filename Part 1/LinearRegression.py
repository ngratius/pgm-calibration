import json
import pickle
import numpy as np
from pathlib import Path
import sklearn.linear_model

from Model import Model

class CappedLinearRegression(Model):
    def __init__(self, low_cap = 0.0, high_cap = 1.0):
        self.low_cap = low_cap
        self.high_cap = high_cap
        self.model = sklearn.linear_model.LinearRegression(fit_intercept=True)

    def load_data(self, X : np.ndarray, y : np.ndarray):
        self.X = np.array(X).reshape(-1, 1)
        self.y = np.array(y).reshape(-1, 1)

    def train(self):
        self.model = self.model.fit(self.X, self.y)
    
    def params(self):
        return (self.model.coef_[0,0], self.model.intercept_[0])
    
    def save(self, name, outdir='.'):
        with open(Path(outdir, name).with_suffix(".pkl"), 'wb') as f:
            pickle.dump(self.model, f)

    def predict(self, X):
        return np.minimum(np.maximum(self.model.predict(X), self.low_cap), self.high_cap)

    def load(self, name, dir):
        print(Path(dir, name).with_suffix(".pkl"))
        with open(Path(dir, name).with_suffix(".pkl"), 'rb') as f:
            self.model = pickle.load(f)

    def set_params(self, params):
        self.model.coef_ = np.array([[params[0]]])
        self.model.intercept_ = np.array([params[1]])


