import numpy as np
from dataset_parsing import simulations_dataset as ds

def load_all_data():
    datasets = []
    # for simulation_number in range(1, 96):
    for simulation_number in [15,20,2]:
        if simulation_number == 25 or simulation_number == 44 or simulation_number == 78:
            continue
        datasets.append((f"Sim{simulation_number}", ds.get_dataset_simulation(simNr=simulation_number)))

    return datasets



