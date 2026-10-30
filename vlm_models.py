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


class CaptionDecoder(nn.Module):
    """Small decoder-only transformer that generates a caption from an image embedding.

    The image embedding is projected to one "visual token" which is put in front of the caption tokens:
        input:  [IMG] <bos> red   circle
        output:   -   red   circle <eos>
    A causal mask makes sure that every position only sees the image and the tokens before it.
    """

    def __init__(self, vocab_size, image_dim, d_model=64, n_heads=4, n_layers=2,
                 max_length=8, dropout=0.1, padding_idx=0):
        """
        :param vocab_size: number of tokens in the vocabulary (incl. <bos> and <eos>)
        :param image_dim: dimension of the incoming image embedding
        :param d_model: width of the transformer
        :param n_heads: number of attention heads
        :param n_layers: number of transformer layers
        :param max_length: maximum number of caption tokens (incl. <bos>)
        :param dropout: dropout probability inside the transformer
        :param padding_idx: index of the <pad> token
        """
        super().__init__()
        self.image_projection = nn.Linear(image_dim, d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        # +1 for the image token in front of the caption
        self.position_embedding = nn.Embedding(max_length + 1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        # A decoder-only transformer is a stack of self-attention layers with a causal mask.
        # Without cross-attention, nn.TransformerEncoder is exactly that.
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        self.max_length = max_length

    def forward(self, image_features, input_ids):
        """Returns next-token logits of shape (B, T, vocab_size) for input_ids of shape (B, T)."""
        image_token = self.image_projection(image_features).unsqueeze(1)
        x = torch.cat([image_token, self.token_embedding(input_ids)], dim=1)
        positions = torch.arange(x.size(1), device=x.device)
        x = x + self.position_embedding(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        # Drop the image position: output i predicts the token following input_ids[:, i]
        return self.output(self.norm(x[:, 1:]))

    @torch.no_grad()
    def generate(self, image_features, bos_id, eos_id):
        """Greedy decoding. Returns generated token ids (B, <= max_length) without the <bos> token."""
        batch_size = image_features.size(0)
        token_ids = torch.full(
            (batch_size, 1), bos_id, dtype=torch.long, device=image_features.device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=image_features.device)
        for _ in range(self.max_length):
            next_ids = self(image_features, token_ids)[:, -1].argmax(dim=-1)
            # Captions which are already finished just keep on producing <eos>
            next_ids = next_ids.masked_fill(finished, eos_id)
            token_ids = torch.cat([token_ids, next_ids.unsqueeze(1)], dim=1)
            finished |= next_ids == eos_id
            if finished.all():
                break
        return token_ids[:, 1:]
