"""Loss functions for CLIP-style VLM models."""

import torch
import torch.nn.functional as F


def contrastive_loss(image_embeddings, text_embeddings, token_ids, temperature):
    """Symmetric contrastive loss with multiple positives per combination."""
    image_embeddings = F.normalize(image_embeddings, dim=1)
    text_embeddings = F.normalize(text_embeddings, dim=1)
    logits = image_embeddings @ text_embeddings.T / temperature

    # Several images in a batch can have the same caption. They should all be
    # treated as positives rather than as negatives of one another.
    positive_mask = torch.all(
        token_ids.unsqueeze(1) == token_ids.unsqueeze(0), dim=2
    )
    log_positive_image = torch.logsumexp(
        logits.masked_fill(~positive_mask, float('-inf')), dim=1
    )
    log_all_image = torch.logsumexp(logits, dim=1)
    image_to_text = -(log_positive_image - log_all_image).mean()

    log_positive_text = torch.logsumexp(
        logits.T.masked_fill(~positive_mask.T, float('-inf')), dim=1
    )
    log_all_text = torch.logsumexp(logits.T, dim=1)
    text_to_image = -(log_positive_text - log_all_text).mean()
    return (image_to_text + text_to_image) / 2
