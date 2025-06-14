# src/data_loader.py

import pandas as pd
import os

def load_data(data_dir="/Volumes/Extreme SSD/calorieEstimator/data/data"):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return train, test