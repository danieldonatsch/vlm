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


class SpatialImageEncoder(nn.Module):
    """CNN encoder which returns a grid of feature vectors ("visual tokens") instead of one vector.

    conv0-2 are identical to ImageEncoder, so their weights can be loaded from an ImageEncoder checkpoint.
    The fully connected layer is replaced by a fourth conv layer and an average pooling to a fixed grid,
    so the encoder works for any (large enough) image size.
    """

    # Layers shared with ImageEncoder
    PRETRAINED_LAYERS = ('conv0', 'bn0', 'conv1', 'bn1', 'conv2', 'bn2')

    def __init__(self, img_channels=3, out_channels=64, grid_size=4):
        """
        :param img_channels: number of image channels (1 for grayscale, 3 for color)
        :param out_channels: dimension of each visual token
        :param grid_size: the output grid has grid_size x grid_size tokens
        """
        super().__init__()
        self.conv0 = nn.Conv2d(img_channels, 8, 5, 1, padding=0)
        self.bn0 = nn.BatchNorm2d(8)
        self.conv1 = nn.Conv2d(8, 16, 5, 2, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 5, 2, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, out_channels, 5, 2, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.pool = nn.AdaptiveAvgPool2d(grid_size)
        self.latent_dim = out_channels
        self.n_tokens = grid_size * grid_size
        self.pretrained_layers_frozen = False

    def load_image_encoder_weights(self, state_dict):
        """Loads conv0-2 (and their BatchNorms) from an ImageEncoder state dict; fc1 is ignored."""
        state_dict = {
            name: value for name, value in state_dict.items()
            if name.split('.')[0] in self.PRETRAINED_LAYERS
        }
        self.load_state_dict(state_dict, strict=False)

    def freeze_pretrained_layers(self):
        """Stops training of conv0-2. Their BatchNorms stay in eval mode, see train()."""
        for name in self.PRETRAINED_LAYERS:
            getattr(self, name).requires_grad_(False)
        self.pretrained_layers_frozen = True
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.pretrained_layers_frozen:
            for name in self.PRETRAINED_LAYERS:
                getattr(self, name).eval()
        return self

    def token_grid_extent(self, image_size) -> tuple:
        """Returns the pixel range (start, end), in x and in y, to which the grid of visual tokens corresponds.

        Each token is placed at the center of its receptive field. Because the convs use no padding,
        the token grid does not cover the image evenly: e.g. for 128 px images and a 4x4 grid the token
        centers are at about 24, 48, 72 and 96 px.
        """
        # Center of the first conv output (offset) and distance between neighboring outputs (jump), in pixels
        offset, jump, size = 0.0, 1, image_size
        for conv in (self.conv0, self.conv1, self.conv2, self.conv3):
            kernel_size, stride = conv.kernel_size[0], conv.stride[0]
            offset += (kernel_size - 1) / 2 * jump
            jump *= stride
            size = (size - kernel_size) // stride + 1
        # AdaptiveAvgPool2d averages bin j over the conv outputs floor(j * size / grid) to ceil((j + 1) * size / grid)
        grid_size = self.pool.output_size if isinstance(self.pool.output_size, int) else self.pool.output_size[0]
        centers = []
        for j in range(grid_size):
            start, end = (j * size) // grid_size, -((-(j + 1) * size) // grid_size)
            # + 0.5: center of the pixel
            centers.append(offset + 0.5 + jump * (start + end - 1) / 2)
        spacing = centers[1] - centers[0] if grid_size > 1 else size * jump
        return centers[0] - spacing / 2, centers[-1] + spacing / 2

    def forward(self, x):
        """Returns visual tokens of shape (B, grid_size * grid_size, out_channels)."""
        for conv, bn in ((self.conv0, self.bn0), (self.conv1, self.bn1),
                         (self.conv2, self.bn2), (self.conv3, self.bn3)):
            x = nn.functional.leaky_relu(bn(conv(x)))
        x = self.pool(x)
        # (B, C, H, W) -> (B, H * W, C), tokens ordered row by row
        return x.flatten(2).transpose(1, 2)


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

    Several visual tokens (e.g. from SpatialImageEncoder) are handled the same way: [IMG_1] ... [IMG_N] <bos> ...
    The visual tokens see each other, the caption tokens see all visual tokens and the tokens before them.
    """

    def __init__(self, vocab_size, image_dim, d_model=64, n_heads=4, n_layers=2,
                 max_length=8, dropout=0.1, padding_idx=0, n_image_tokens=1):
        """
        :param vocab_size: number of tokens in the vocabulary (incl. <bos> and <eos>)
        :param image_dim: dimension of the incoming image embedding
        :param d_model: width of the transformer
        :param n_heads: number of attention heads
        :param n_layers: number of transformer layers
        :param max_length: maximum number of caption tokens (incl. <bos>)
        :param dropout: dropout probability inside the transformer
        :param padding_idx: index of the <pad> token
        :param n_image_tokens: number of visual tokens per image
        """
        super().__init__()
        self.image_projection = nn.Linear(image_dim, d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        # The image tokens in front of the caption also get a learned position embedding. For a grid of
        # visual tokens, every grid cell has its own index, so this also tells the decoder where a token is.
        self.position_embedding = nn.Embedding(max_length + n_image_tokens, d_model)
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
        self.n_image_tokens = n_image_tokens

    def embed(self, image_features, input_ids) -> tuple:
        """Returns the transformer input (visual tokens followed by caption tokens) and the attention mask.

        image_features has shape (B, D) for one visual token or (B, n_image_tokens, D).
        """
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)
        image_tokens = self.image_projection(image_features)
        x = torch.cat([image_tokens, self.token_embedding(input_ids)], dim=1)
        positions = torch.arange(x.size(1), device=x.device)
        x = x + self.position_embedding(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        # The visual tokens may attend to each other (for one token this changes nothing)
        mask[:self.n_image_tokens, :self.n_image_tokens] = 0.0
        return x, mask

    def forward(self, image_features, input_ids):
        """Returns next-token logits of shape (B, T, vocab_size) for input_ids of shape (B, T).

        image_features has shape (B, D) for one visual token or (B, n_image_tokens, D).
        """
        x, mask = self.embed(image_features, input_ids)
        x = self.transformer(x, mask=mask)
        # Drop the image positions: output i predicts the token following input_ids[:, i]
        return self.output(self.norm(x[:, self.n_image_tokens:]))

    @torch.no_grad()
    def attention_maps(self, image_features, input_ids) -> list:
        """Returns the attention weights of every layer, for analysis only (call in eval mode).

        nn.TransformerEncoderLayer computes its attention with need_weights=False, so the weights are
        computed here a second time, with the same inputs, by calling the layer's attention module directly.

        :return: (list) one tensor per layer of shape (B, n_heads, L, L) with L = n_image_tokens + T.
            Entry [b, h, i, j] is how much position i attends to position j.
        """
        x, mask = self.embed(image_features, input_ids)
        maps = []
        for layer in self.transformer.layers:
            # norm_first=True: the attention sees the normalized layer input
            h = layer.norm1(x)
            _, weights = layer.self_attn(
                h, h, h, attn_mask=mask, need_weights=True, average_attn_weights=False,
            )
            maps.append(weights)
            x = layer(x, src_mask=mask)
        return maps

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
