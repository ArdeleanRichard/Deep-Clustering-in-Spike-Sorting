import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.metrics import v_measure_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt


class AutoClustering(nn.Module):
    def __init__(self, input_dim, n_clusters, alpha_final=500.0, gamma=4.0):
        super(AutoClustering, self).__init__()
        self.alpha_final = alpha_final
        self.gamma = gamma
        self.n_clusters = n_clusters

        # Encoder layers
        self.encoder_layers = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, self.n_clusters)
        )

        # Decoder layers
        self.decoder_layers = nn.Sequential(
            nn.Linear(self.n_clusters, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, input_dim)
        )

    def quasi_max(self, logits, alpha):
        scaled_logits = alpha * logits
        return torch.softmax(scaled_logits, dim=-1)

    def forward(self, x, alpha):
        # Encoder: Input -> Cluster Assignments
        logits = self.encoder_layers(x)
        clusters = self.quasi_max(logits, alpha)

        # Decoder: Cluster Assignments -> Exemplars
        exemplars = self.decoder_layers(clusters)
        return clusters, exemplars

    def alpha_schedule(self, epoch, max_epochs):
        return 1.0 + (self.alpha_final - 1.0) * ((epoch / max_epochs) ** self.gamma)

    def loss_function(self, inputs, exemplars):
        return torch.mean(torch.sum((inputs - exemplars) ** 2, dim=1))

    def fit_predict(self, X, learning_rate=0.001, max_epochs=100, batch_size=32):
        dataset = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                             batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(max_epochs):
            alpha = self.alpha_schedule(epoch, max_epochs)
            epoch_loss = 0.0

            for batch in dataset:
                batch = batch[0]
                optimizer.zero_grad()
                clusters, exemplars = self.forward(batch, alpha)
                loss = self.loss_function(batch, exemplars)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {epoch_loss:.4f}, Alpha: {alpha:.2f}")

        with torch.no_grad():
            final_clusters, _ = self.forward(torch.tensor(X, dtype=torch.float32), self.alpha_final)
            predicted_labels = torch.argmax(final_clusters, dim=1).numpy()

        return predicted_labels

    def plot(self, X, ground_truth):
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

        with torch.no_grad():
            final_clusters, exemplars = self.forward(torch.tensor(X, dtype=torch.float32), self.alpha_final)
            predicted_labels = torch.argmax(final_clusters, dim=1).numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Ground truth clusters
        axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=ground_truth, cmap='viridis', s=50, alpha=0.7)
        axes[0].set_title("Ground Truth")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")

        # Predicted clusters
        exemplars_2d = pca.transform(np.unique(exemplars.numpy(), axis=0))
        axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=predicted_labels, cmap='viridis', s=50, alpha=0.7)
        axes[1].scatter(exemplars_2d[:, 0], exemplars_2d[:, 1], c='red', s=100, alpha=0.7)
        axes[1].set_title("AutoClustering")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")

        plt.tight_layout()
        plt.show()

        print("F:", v_measure_score(ground_truth, predicted_labels))


if __name__ == "__main__":
    data = load_iris()
    ground_truth = data.target
    X = StandardScaler().fit_transform(data.data)

    print(X.shape)

    model = AutoClustering(X.shape[1], 3)
    predicted_labels = model.fit_predict(X)
    model.plot(X, ground_truth)
