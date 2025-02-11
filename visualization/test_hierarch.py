import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Parameters
num_algorithms = 50
num_datasets = 7
num_scores = 6

# Generate random data (shape: algorithms x datasets x scores)
data = np.random.rand(num_algorithms, num_datasets, num_scores)

def plot_hierarchical_visualization(data, num_algorithms, num_datasets, num_scores):
    fig, ax = plt.subplots(figsize=(16, 11))

    # Overall grid dimensions, with spacing
    total_width = 1.0
    total_height = 1.0

    algo_height = total_height / (num_algorithms + num_algorithms - 1)  # Include spacing rows
    dataset_width = total_width / (num_datasets + (num_datasets - 1) * (1 / num_scores))  # Smaller spacing columns

    spacing_width = dataset_width / num_scores  # Spacing size equivalent to one score width

    # Loop through algorithms and datasets
    for algo_idx in range(num_algorithms):
        for dataset_idx in range(num_datasets):
            # Top-left corner of the dataset block
            x0 = dataset_width * dataset_idx + spacing_width * dataset_idx
            y0 = total_height - algo_height * (2 * algo_idx + 1)

            # Draw the dataset block
            ax.add_patch(Rectangle((x0, y0), dataset_width, algo_height, edgecolor='black', facecolor='none'))

            # Draw small rectangles for validation scores inside the dataset block
            score_width = dataset_width / num_scores
            for score_idx in range(num_scores):
                score_x0 = x0 + score_width * score_idx
                score_y0 = y0

                # Map the score value to a color
                color = plt.cm.viridis(data[algo_idx, dataset_idx, score_idx])
                ax.add_patch(Rectangle((score_x0, score_y0), score_width, algo_height, color=color))

    # Add labels and formatting
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_height)
    ax.set_xticks([((dataset_width + spacing_width) * i + 0.5 * dataset_width) for i in range(num_datasets)])
    ax.set_xticklabels([f"Dataset {i+1}" for i in range(num_datasets)])
    ax.set_yticks([(total_height - (2 * i + 0.5) * algo_height) for i in range(num_algorithms)])  # Align ticks with rows
    ax.set_yticklabels([f"Algorithm {i+1}" for i in range(num_algorithms)])
    ax.set_title("Comparison")
    ax.set_xlabel("Datasets")
    ax.set_ylabel("Algorithms")



    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("Validation Score")



    # Add the score ticks on top of the colored squares
    # Define positions for the top ticks, above the colored blocks
    top_ticks_x = []
    top_tick_labels = []
    for i in range(num_datasets):
        for score_idx in range(num_scores):
            top_ticks_x.append((dataset_width + spacing_width) * i + score_idx * (dataset_width / num_scores) + 0.5 * (
                        dataset_width / num_scores))
            top_tick_labels.append(f"S{score_idx + 1}")



    # Create a second x-axis on top with different labels (if done before colorbar, it moves it)
    ax2 = ax.twiny()
    ax2.set_xticks(top_ticks_x)
    ax2.set_xticklabels(top_tick_labels)

    # Adjust the position of the second x-axis
    ax2.xaxis.set_ticks_position('top')

    plt.tight_layout()
    plt.show()

# Plot the hierarchical visualization
plot_hierarchical_visualization(data, num_algorithms, num_datasets, num_scores)