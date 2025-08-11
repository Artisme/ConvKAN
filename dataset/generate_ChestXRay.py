import sys
import pandas as pd
import numpy as np
import os
import random
import torch
import torchvision.transforms as transforms
from sklearn.utils import resample, shuffle
from torch.utils.data import DataLoader
from PIL import Image

# Add the parent directory to the path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.utils.dataset_utils import check, separate_data, split_data, save_file

# --- Configuration based on the paper ---
random.seed(1)
np.random.seed(1)
num_clients = 10
num_classes = 2
# The paper specifies an input resolution of 128x128
img_size = 128

# --- Paths ---
# Use a new directory to store the dataset generated with the paper's settings
dir_path = "ChestXRay/" 
data_path = "ChestXRay/raw/" 

def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train/")
    test_path = os.path.join(dir_path, "test/")

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    if not os.path.exists(data_path):
        print(f"Error: Raw data directory not found at {data_path}")
        return

    all_files = []
    for split in ['train', 'test', 'val']:
        split_path = os.path.join(data_path, split)
        if not os.path.exists(split_path):
            continue
        
        for label_name in ['NORMAL', 'PNEUMONIA']:
            class_num = 0 if label_name == 'NORMAL' else 1
            folder_path = os.path.join(split_path, label_name)
            
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.jpeg'):
                    all_files.append({'file_name': file_name, 'class': class_num, 'image_folder': folder_path})
    
    df = pd.DataFrame(all_files)
    df = shuffle(df)

    # Note: The paper does not specify dataset balancing. We are keeping the original distribution.
    print(f"Total images found: {len(df)}")
    print(f"Class counts:\n{df['class'].value_counts()}")

    # --- Preprocessing transformations based on the paper ---
    # The paper mentions resizing to 128x128 and normalization.
    # It also mentions on-the-fly data augmentation (flips, rotations).
    # This would typically be added to the training DataLoader, not here.
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # Standard normalization
    ])
    
    dataset_image = []
    dataset_label = []
    print("Loading and preprocessing all images...")
    for index, row in df.iterrows():
        img_path = os.path.join(row['image_folder'], row['file_name'])
        image = Image.open(img_path)
        tensor_img = transform(image)
        dataset_image.append(tensor_img)
        dataset_label.append(row['class'])

    dataset_image = torch.stack(dataset_image).numpy()
    dataset_label = np.array(dataset_label, dtype=np.int32)
    
    print(f'Total data loaded into memory: {len(dataset_image)}')

    X, y, statistic = separate_data(
        (dataset_image, dataset_label), 
        num_clients, 
        num_classes,  
        niid=niid, 
        balance=balance, 
        partition=partition
    )
    
    train_data, test_data = split_data(X, y)
    
    save_file(
        config_path, 
        train_path, 
        test_path, 
        train_data, 
        test_data, 
        num_clients, 
        num_classes, 
        statistic, 
        niid, 
        balance, 
        partition
    )


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)
