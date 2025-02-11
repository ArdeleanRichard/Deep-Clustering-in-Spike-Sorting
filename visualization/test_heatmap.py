import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Parameters
num_algorithms = 50
num_datasets = 7
num_scores = 6

# Generate random data (shape: algorithms x datasets x scores)
data = np.random.rand(num_algorithms, num_datasets, num_scores)

# Create heatmaps for each score
for score_idx in range(num_scores):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data[:, :, score_idx], annot=False, cmap="viridis",
                xticklabels=[f"Dataset {i + 1}" for i in range(num_datasets)],
                yticklabels=[f"Algorithm {i + 1}" for i in range(num_algorithms)])

    plt.title(f"Heatmap for Validation Score {score_idx + 1}")
    plt.xlabel("Datasets")
    plt.ylabel("Algorithms")
    plt.tight_layout()
    plt.show()
