import numpy as np
from dataset_parsing import simulations_dataset as ds

def load_all_data():
    datasets = [
        ("Sim4", ds.get_dataset_simulation(simNr=4)),
    ]

    return datasets



