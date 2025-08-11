import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_comparison(args):
    """
    Reads a specific metric from two HDF5 files (cnn.h5 and kan.h5)
    and plots them on a single graph for comparison.
    """
    # Define the model names and corresponding filenames
    models = {
        "CNN": f"{args.dataset}_FedAvg_CNN_{args.goal}_{args.time}.h5",
        "KAN": f"{args.dataset}_FedAvg_KAN_{args.goal}_{args.time}.h5"
    }
    
    # Store the data for each model
    plot_data = {}

    # --- Read data from both H5 files ---
    for model_name, filename in models.items():
        # Construct the path to the results directory (one level up from the script location)
        file_path = os.path.join(os.path.dirname(__file__), "..", "results", filename)

        if not os.path.exists(file_path):
            print(f"Warning: Results file for {model_name} not found at '{file_path}'. Skipping.")
            continue

        try:
            with h5py.File(file_path, 'r') as hf:
                if args.metric not in hf.keys():
                    print(f"Error: Metric '{args.metric}' not found in {file_path}")
                    print(f"Available metrics in this file are: {list(hf.keys())}")
                    continue
                
                # Read the metric values and store them
                values = hf[args.metric][()]
                plot_data[model_name] = values

        except Exception as e:
            print(f"Error reading or parsing the H5 file for {model_name}: {e}")

    if len(plot_data) < 2:
        print("\nError: Could not load data for both models. Aborting plot generation.")
        return

    # --- Create the plot ---
    plt.figure(figsize=(12, 7))
    
    # Plot data for each model that was successfully loaded
    for model_name, values in plot_data.items():
        rounds = np.arange(len(values))
        plt.plot(rounds, values, marker='o', linestyle='-', markersize=4, label=f'{model_name}')

    # --- Formatting ---
    metric_name = args.metric.replace('rs_', '').replace('_', ' ').title()
    title = f"Comparison of {list(plot_data.keys())[0]} vs. {list(plot_data.keys())[1]} on {args.dataset}\n({metric_name} vs. Global Rounds)"
    
    plt.title(title, fontsize=16)
    plt.xlabel("Global Round", fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # --- Save the Plot ---
    model_names_str = "_vs_".join(plot_data.keys())
    plot_filename = f"{args.dataset}_{model_names_str}_{args.metric}_comparison_run{args.time}.png"
    save_path = os.path.join(os.path.dirname(__file__), "..", "results", plot_filename)
    plt.savefig(save_path, dpi=300)
    print(f"\nComparison plot successfully saved to: {save_path}")
    
    # --- Show the Plot ---
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a comparison of two models from PFLlib's HDF5 output.")
    parser.add_argument('-data', "--dataset", type=str, required=True, 
                        help="Base name of the dataset (e.g., ChestXRay)")
    parser.add_argument('-metric', "--metric", type=str, required=True, 
                        choices=["rs_test_acc", "rs_train_loss", "rs_test_auc"],
                        help="Metric to plot from the H5 files.")
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for the experiment (e.g., test).")
    parser.add_argument('-t', "--time", type=int, default=0,
                        help="The specific run number to plot (e.g., 0).")
    
    args = parser.parse_args()
    plot_comparison(args)