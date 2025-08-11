# Federated Learning with Conv-KAN for Medical Imaging

This repository contains the official implementation for the research paper, **"Federated Learning with Convolutional Kolmogorov-Arnold Network in Healthcare."** It provides a framework for training and evaluating both novel Conv-KAN architectures and standard CNNs in a simulated, privacy-preserving federated learning environment.

The primary goal of this project is to benchmark the performance of Convolutional Kolmogorov-Arnold Networks (Conv-KANs) against traditional CNNs for pneumonia classification on the Chest X-Ray dataset.

-----

## Key Features

  * **Novel Architectures**: Includes implementations for `ConvKAN` and a baseline `FedLightCNN`.
  * **Federated Algorithms**: Supports standard `FedAvg` and the more robust `FedProx` to handle client drift.
  * **Comprehensive Evaluation**: The framework is equipped to calculate and report a full suite of metrics: **Accuracy, AUC-ROC, Precision, Recall, F1-Score, and Specificity**.
  * **Reproducibility**: The code is structured to allow for easy reproduction of the experiments described in the paper.

-----

## How to Run

Follow these steps to set up the environment, download the data, and run the training simulations.

### 1\. Clone and Set Up the Environment

First, clone the repository and create a Conda environment from the provided `environment.yml` file.

```bash
git clone https://github.com/Artisme/ConvKAN
cd ConvKAN
conda env create -f environment.yml
conda activate pfllib
```

### 2\. Download and Set Up the Dataset (Manual)

This project uses the public Chest X-Ray (Pneumonia) dataset from Kaggle.

  * **Download**: Go to the [Kaggle dataset page](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and download the files.
  * **Unzip**: Unzip the downloaded file (`archive.zip`). This will create a folder named `chest_xray`.
  * **Organize Folders**: Create the necessary directory structure and move the data into it. The final structure should look like this:
    ```
    PFLlib/
    └── dataset/
        └── ChestXRay/
            └── raw/
                ├── train/
                ├── test/
                └── val/
    ```
    You can create these directories and move the `chest_xray` contents with the following commands from inside the `PFLlib/dataset/` directory:
    ```bash
    # From within the PFLlib/dataset/ directory
    mkdir -p ChestXRay/raw

    # Assuming the unzipped 'chest_xray' folder is in your Downloads
    mv ~/Downloads/chest_xray/* ChestXRay/raw/
    ```

### 3\. Generate the Federated Dataset

After setting up the raw images, you need to partition them into a federated dataset for the clients.

  * **Run Generation Script**: Navigate to the `dataset` directory and run the generation script. This will create a `ChestXRay_Paper/` folder containing the data partitioned for 10 clients, as specified in the paper.
    ```bash
    cd dataset
    python generate_ChestXRay.py iid - -
    ```

### 4\. Run a Training Simulation

You are now ready to run the experiments. All training is launched from the `system` directory.

  * **Navigate to the system directory**:
    ```bash
    cd ../system
    ```
  * **Run the ConvKAN experiment**:
    ```bash
    python main.py -data ChestXRay_Paper -m ConvKAN -algo FedAvg -gr 50 -ncl 2 -nc 10 -jr 0.5 -ls 5 -lbs 32 -opt adam -lr 0.0001
    ```
  * **Run the FedLightCNN baseline experiment**:
    ```bash
    python main.py -data ChestXRay_Paper -m FedLightCNN -algo FedAvg -gr 50 -ncl 2 -nc 10 -jr 0.5 -ls 5 -lbs 32 -opt adam -lr 0.0001
    ```

### 5\. Plot the Results

After a simulation is complete, a `.h5` file containing the results will be saved in the `results/` directory. Use the `plot_results.py` script to visualize the performance.

  * **Plot the test accuracy for the FedLightCNN run**:
    ```bash
    python plot_results.py -data ChestXRay_Paper -algo FedAvg -metric rs_test_acc
    ```
