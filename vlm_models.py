import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, img_channels=1, img_size=28, latent_dims=128):
        """CNN encoder for square images.

        :param img_channels: number of image channels (1 for grayscale, 3 for color)
        :param img_size: number of pixels on one side of the square image
        :param latent_dims: dimension of the encoder output
        """
        super().__init__()
        self.conv0 = nn.Conv2d(img_channels, 8, 5, 1, padding=0)
        self.bn0 = nn.BatchNorm2d(8)
        self.conv1 = nn.Conv2d(8, 16, 5, 2, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 5, 2, padding=0)
        self.bn2 = nn.BatchNorm2d(32)

        spatial_dim = (img_size / 4 - 4)
        n_conv_out_vals = 32 * int(spatial_dim * spatial_dim)
        self.fc1 = nn.Linear(n_conv_out_vals, latent_dims)
        self.latent_dim = latent_dims

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.leaky_relu(x)
        x = torch.flatten(x, 1)
        return self.fc1(x)


class ProjectionHead(nn.Module):
    """MLP used to map encoder representations to contrastive space."""

    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class LinearProjectionHead(nn.Module):
    """Linear projection into a shared multimodal embedding space."""

    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class EmbeddingClassifier(nn.Module):
    """Small classifier operating on frozen image embeddings."""

    def __init__(self, input_dim, hidden_dim=64, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, embeddings):
        return self.net(embeddings)


class LinearEmbeddingClassifier(nn.Module):
    """Linear probe for evaluating the separability of image embeddings."""

    def __init__(self, input_dim, n_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_classes)

    def forward(self, embeddings):
        return self.linear(embeddings)


class TextEncoder(nn.Module):
    """Simple text encoder using token embeddings and mean pooling."""

    def __init__(self, vocab_size, embedding_dim=64, padding_idx=0):
        super().__init__()
        self.token_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )

    def forward(self, token_ids):
        token_vectors = self.token_embedding(token_ids)
        non_padding = token_ids != self.token_embedding.padding_idx
        non_padding = non_padding.unsqueeze(-1)
        summed_vectors = (token_vectors * non_padding).sum(dim=1)
        token_counts = non_padding.sum(dim=1).clamp_min(1)
        return summed_vectors / token_counts
