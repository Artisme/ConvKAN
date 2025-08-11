import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_graph(args):
    """
    Reads simulation results from a specific HDF5 (.h5) file and generates a plot.
    """
    # Construct the path to the results directory, which is one level up from system/
    base_results_dir = os.path.join(os.path.dirname(__file__), "..", "results")

    # Construct the exact filename based on the user's provided format
    # e.g., ChestXRay_Paper_FedAvg_test_0.h5
    filename = f"{args.dataset}_{args.algorithm}_{args.goal}_{args.time}.h5"
    file_path = os.path.join(base_results_dir, filename)

    if not os.path.exists(file_path):
        print(f"Error: Results file not found at '{file_path}'")
        print("Please ensure the dataset, algorithm, goal, and time arguments are correct.")
        return

    try:
        # Open the HDF5 file in read-only mode
        with h5py.File(file_path, 'r') as hf:
            # Check if the requested metric exists as a key in the file
            if args.metric not in hf.keys():
                print(f"Error: Metric '{args.metric}' not found in {file_path}")
                print(f"Available metrics in this file are: {list(hf.keys())}")
                return
            
            # Read the data for the specified metric
            values = hf[args.metric][()]
            rounds = np.arange(len(values))

    except Exception as e:
        print(f"Error reading or parsing the H5 file: {e}")
        return

    # Create the plot
    plt.figure(figsize=(12, 7))
    plt.plot(rounds, values, marker='o', linestyle='-', markersize=4, label=f'{args.algorithm}')
    
    # --- Formatting ---
    metric_name = args.metric.replace('rs_', '').replace('_', ' ').title()
    # title = f"Global Model {metric_name} vs. Communication Rounds\n(Run #{args.time} - {args.algorithm} on {args.dataset})"
    title = f"Global Model {metric_name} vs. Communication Rounds"
    
    plt.title(title, fontsize=16)
    plt.xlabel("Global Round", fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    # --- Save the Plot ---
    plot_filename = f"{args.dataset}_{args.algorithm}_{args.metric}_run{args.time}_plot.png"
    save_path = os.path.join(base_results_dir, plot_filename)
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot successfully saved to: {save_path}")
    
    # --- Show the Plot ---
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot simulation results from PFLlib's HDF5 output.")
    parser.add_argument('-data', "--dataset", type=str, required=True, 
                        help="Name of the dataset (e.g., ChestXRay_Paper)")
    parser.add_argument('-algo', "--algorithm", type=str, required=True, 
                        help="Name of the algorithm (e.g., FedAvg)")
    parser.add_argument('-metric', "--metric", type=str, required=True, 
                        choices=["rs_test_acc", "rs_train_loss", "rs_test_auc"],
                        help="Metric to plot from the H5 file.")
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for the experiment (e.g., test).")
    parser.add_argument('-t', "--time", type=int, default=0,
                        help="The specific run number to plot (e.g., 0).")
    
    args = parser.parse_args()
    plot_graph(args)

