# pgm-calibration

#### Spring 2023 10-708 Final Project
#### Sudeep Agarwal, Adiel Felsen, Nicolas Gratius

## Part 1: Simulation Accuracy Analysis

To run all commands, first `cd "Part\ 1"`

### Generate Synthetic Data

`python3 TrialData.py`

### Running Simulation

`python3 Simulator.py`

## Part 2: Computational Cost Analysis

To run all commands, first `cd "Part\ 2"`

### Visualizing a Randomly Generated DAG

`python3 dag.py`

Result is saved under `Part\ 2/results/graph_structure.png`

### Generating Accuracy Plots

`python3 measure_dag_acc.py --method <method>`

Method can be:

* 'e' : Exact Inference
    * Outputs a dictionary of all states following exact inference
* 'l' : Likelihood Weighted Approximation Approximation
    * Shows a plot of accuracy of likelihood weighting approximation compared to exact inference against number of samples
* 'r' : Rejection Approximation
    * Shows a plot of accuracy of rejection sampling approximation compared to exact inference against number of samples
* 'f' : Forward Approximation
    * Shows a plot of accuracy of forward sampling approximation compared to exact inference against number of samples

### Generating Performance Plots

`python3 measure_dag_performance.py --method <method>`

Plots performance and memory requirements as a function of graph size (number of nodes and variable cardinality)

Method can be:
* 'e' : Exact Inference

* 'l' : Likelihood Weighted Approximation

* 'r' : Rejection Approximation
    * Note: Rejection sampling can easily get stuck (long inference times) and requires a restart.

* 'f' : Forward Approximation


### Results

Accuracy plots can be found at `Part\ 2/results/accuracy/*.png`

Time and memory complexity plots can be found at `Part\ 2/results/time_complexity/*.png`

Graph structure visualization can be found at 
`Part\ 2/results/graph_structure.png`

Conditional probability table can be found at `Part\ 2/results/cpd.png`
