from dag import ParameterModel
from time import process_time
import tracemalloc
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import argparse

matplotlib.style.use('ggplot')

def simulate(n, c, n_samples=None, method="exact"):

    model = ParameterModel(num_nodes=n, node_cardinality=c)
    model.construct_pgmpy()

    # Measure time and memory taken for inference
    tracemalloc.start()
    start_time = process_time() 

    if method == 'exact':
        states = model.exact_infer(evidence_var='h_0', evidence_state=0, show_progress=False)
    elif method == 'gibbs_sampling':
        states = model.gibbs_infer(evidence_var='h_0', evidence_state=0, show_progress=False)
    elif method == 'likelihood_weighted':
        approx_samples = model.likelihood_weighted_sample("h_0", 0, show_progress=False, n_samples=n_samples)
        states = model.sample_inference(approx_samples, nodes=[n.name for n in model.all_nodes])
    elif method == 'rejection':
        approx_samples = model.rejection_sample("h_0", 0, show_progress=False, n_samples=n_samples)
        states = model.sample_inference(approx_samples, nodes=[n.name for n in model.all_nodes])
    elif method == 'forward':
        approx_samples = model.forward_sample("h_0", 0, show_progress=False, n_samples=n_samples)
        states = model.sample_inference(approx_samples, nodes=[n.name for n in model.all_nodes])
    else:
        raise Exception("Invalid inference method")

    end_time = process_time()
    memory = tracemalloc.get_traced_memory()
    memory_diff = (memory[1] - memory[0])/(1024**2)
    
    tracemalloc.stop()
    
    duration = end_time - start_time

    print(f"{duration = }")
    print(f"{memory_diff = } MB")

    del model

    return duration, memory_diff


def plot(ns, cs, ns_duration, cs_duration, ns_memory_usage, cs_memory_usage, method):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    

    axs0_2 = axs[0].twinx()
    axs[0].plot(ns, ns_duration, label="Inference Time", color="#fa4b48")
    axs[0].set_ylabel("Inference Time (s)", color="#fa4b48")
    axs[0].set_xlabel("Number of Nodes (Fixed cardinality = 3)")
    axs[0].set_xticks(ns)
    axs[0].set_title("Number of Nodes vs. Computational Cost")
    
    axs0_2.plot(ns, ns_memory_usage, label="Memory Usage", color="#0ca1cf")
    axs0_2.set_ylabel("Memory Usage (MB)", color="#0ca1cf")
    
    axs[1].plot(cs, cs_duration, label="Inference Time", color="#fa4b48")
    axs[1].set_ylabel("Inference Time (s)", color="#fa4b48")
    axs[1].set_xlabel("Node Cardinality (Fixed node count = 5)")
    axs[1].set_xticks(cs)
    axs[1].set_title("Node Cardinality vs. Computational Cost")
    
    axs1_2 = axs[1].twinx()
    axs1_2.plot(cs, cs_memory_usage, label="Memory Usage", color="#0ca1cf")
    axs1_2.set_ylabel("Memory Usage (MB)", color="#0ca1cf")
    # axs[0].set_xlim(0, 9)

    plt.suptitle(f'Performance of {method} inference')
    plt.tight_layout()
    plt.show()

def plot_all(method="exact", n_samples=1000, min_n=10, max_n=20, min_c=2, max_c=10):
    ns = np.arange(min_n, max_n+2, 2)
    cs = np.arange(min_c, max_c+1, 1)
    
    ns_duration = []
    ns_memory_usage = []
    for n in ns:
        print(f"{n=}")
        c = 3
        duration, memory_usage = simulate(n, c, n_samples=n_samples, method=method)
        ns_duration.append(duration)
        ns_memory_usage.append(memory_usage)

    cs_duration = []
    cs_memory_usage = []
    for c in cs:
        print(f"{c=}")
        n = 5
        duration, memory_usage = simulate(n, c, n_samples=n_samples, method=method)
        cs_duration.append(duration)
        cs_memory_usage.append(memory_usage)
    plot(ns=ns, cs=cs, ns_duration=ns_duration, cs_duration=cs_duration, ns_memory_usage=ns_memory_usage, cs_memory_usage=cs_memory_usage, method=method)
            
if __name__ == "__main__":
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
        print("Usage: python3 measure_dag_performance.py --method <method>\n")
        print(show_options(method_map))
        exit()
    try:
        assert args.method in method_map
    except:
        print("Invalid method.")
        print(show_options(method_map))
        exit()

    # Generate the plots
    print(f"Conducting Performace Analysis for: {args.method}")

    plot_all(method=method_map[args.method], n_samples=1000, min_n=10, max_n=20, min_c=2, max_c=9)
    