import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

from dag import ParameterModel

import argparse


def plot_approx_acc(model, method):
    fig, ax = plt.subplots()

    N_TRIALS = 50
    MAX_N_SAMPLES = 10000
    NUM_POINTS = 10
    exact_pred = model.exact_infer("h_0", 0, show_progress=False)
    node_names = [n.name for n in model.all_nodes]

    cis = []
    mean_accs = []
    n_samples = np.geomspace(1, MAX_N_SAMPLES, num=NUM_POINTS, dtype=int)
    for n in n_samples:
        accuracies = []
        for _ in range(N_TRIALS):
            if method == "likelihood_weighted":
                approx_samples = model.likelihood_weighted_sample("h_0", 0, show_progress=False, n_samples=n)
            elif method == "gibbs":
                pass
                # model.gibbs("h_0", 0, show_progress=False, n_samples=n)
            elif method =="rejection":
                approx_samples = model.rejection_sample("h_0", 0, show_progress=False, n_samples=n)
            elif method == "forward":
                approx_samples = model.forward_sample("h_0", 0, show_progress=False, n_samples=n)
            approx_pred = model.sample_inference(approx_samples, node_names)
            accuracies.append(model.approx_vs_exact(approx_predictions= approx_pred, exact_predictions=exact_pred))

        # ci = 1.96*np.std(accuracies)
        mean_acc = np.mean(accuracies)
        cis.append(np.percentile(accuracies, [2.5, 97.5]))
        print(n, np.percentile(accuracies, [2.5, 97.5]))
        # bootstrap = scipy.stats.bootstrap((accuracies,), np.mean, method="percentile")
        # # bs_replicates_heights = accuracy(heights,np.mean,15000)
        # print(bootstrap.confidence_interval)

        # cis.append(bootstrap.confidence_interval)
        mean_accs.append(mean_acc)

        # plt.scatter([n]*len(accuracies), accuracies)


    cis = np.array(cis)
    print(cis)
    mean_accs = np.array(mean_accs)

    
    ax.plot(n_samples, mean_accs)
    ax.fill_between(n_samples, (cis[:,0]), (cis[:,1]), color='b', alpha=.1, label=r"95% CI")

    ax.set_xscale('log')

    plt.title(f"Accuracy of {method} approximate inference")
    plt.xlabel("Number of samples")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
        

if __name__ == "__main__":

    model = ParameterModel(num_nodes=10, node_cardinality=3)

    model.construct_pgmpy()
    

    ### Command line parsing

    def show_options(method_map):
        ret = ""
        ret += "Use one of the following for <method>:\n"
        #show acceptable inputs
        for k, v in method_map.items():
            ret += f"•{k}: {v}\n"
        return ret[:-1]

    #Command line arguments
    method_map = {"e": "exact", "l": "likelihood_weighted", "r": "rejection", "f": "forward"}

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, help=show_options(method_map))
    args = parser.parse_args()

    #Invalid inputs
    if not args.method:
        print("Usage: python3 measure_dag_acc.py --method <method>\n")
        print(show_options(method_map))
        exit()
    try:
        assert args.method in method_map
    except:
        print("Invalid method.")
        print(show_options(method_map))
        exit()

    # Generate the plots
    print(f"Showing accuracy for method: {args.method}")

    if args.method == "e":
        print("Exact:", model.exact_infer("h_0", 0))
    else:
        plot_approx_acc(model, method=method_map[args.method])