"""Helpers shared by the caption training scripts (train_captioner.py, train_multi_captioner.py)."""
import datetime
import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from my_utils import log_message


def encode_images(image_encoder, image_projection, images) -> torch.Tensor:
    """Returns the image features which are fed to the decoder.

    :param image_encoder: (ImageEncoder | SpatialImageEncoder) The image encoder.
    :param image_projection: (LinearProjectionHead | None) CLIP projection head; if None, the raw encoder output is used.
    :param images: (torch.Tensor) Batch of images.
    :return: (torch.Tensor) Image features of shape (B, D), or (B, N, D) for a SpatialImageEncoder.
    """
    features = image_encoder(images)
    if image_projection is not None:
        features = image_projection(features)
    return features


def decode_tokens(token_ids, inverse_vocabulary, eos_id, pad_id) -> list:
    """Turns a batch of token ids into caption strings, stopping at the first <eos>.

    :param token_ids: (torch.Tensor) Token ids of shape (B, T).
    :param inverse_vocabulary: (dict) Mapping index to word.
    :param eos_id: (int) Index of the <eos> token.
    :param pad_id: (int) Index of the <pad> token.
    :return: (list) Captions as strings.
    """
    captions = []
    for row in token_ids.cpu().tolist():
        words = []
        for token_id in row:
            if token_id == eos_id:
                break
            if token_id != pad_id:
                words.append(inverse_vocabulary[token_id])
        captions.append(' '.join(words))
    return captions


def generate_captions(image_encoder, image_projection, decoder, images, vocabulary, inverse_vocabulary, device) -> list:
    """Generates one caption per image with greedy decoding."""
    image_features = encode_images(image_encoder, image_projection, images.to(device))
    token_ids = decoder.generate(image_features, vocabulary['<bos>'], vocabulary['<eos>'])
    return decode_tokens(token_ids, inverse_vocabulary, vocabulary['<eos>'], vocabulary['<pad>'])


def predict_captions(image_encoder, image_projection, decoder, loader, vocabulary, inverse_vocabulary,
                     device) -> tuple:
    """Generates captions for all images of a loader.

    :return: (tuple) true captions and generated captions, both as lists in the order of the loader.
    """
    image_encoder.eval()
    decoder.eval()
    if image_projection is not None:
        image_projection.eval()
    all_targets, all_predictions = [], []
    with torch.no_grad():
        for images, token_ids in loader:
            all_predictions.extend(generate_captions(
                image_encoder, image_projection, decoder, images,
                vocabulary, inverse_vocabulary, device,
            ))
            # Strip <bos>/<eos> from the true captions
            all_targets.extend(decode_tokens(
                token_ids[:, 1:], inverse_vocabulary, vocabulary['<eos>'], vocabulary['<pad>']
            ))
    return all_targets, all_predictions


def train_epoch(image_encoder, image_projection, decoder, loader, optimizer, pad_id, train_encoder, device) -> float:
    """Trains the decoder (and the encoder if train_encoder is True) for one epoch.

    :param train_encoder: (bool) If False, the encoder is kept in eval mode and gets no gradients.
    :return: (float) mean loss per sample
    """
    decoder.train()
    # A frozen encoder stays in eval mode so that its BatchNorm statistics are not changed
    image_encoder.train(train_encoder)
    if image_projection is not None:
        image_projection.train(train_encoder)
    total_loss, n_samples = 0.0, 0

    for images, token_ids in loader:
        images = images.to(device)
        token_ids = token_ids.to(device)
        # Teacher forcing: input is [<bos>, w1, w2], target is [w1, w2, <eos>]
        input_ids = token_ids[:, :-1]
        target_ids = token_ids[:, 1:]
        with torch.set_grad_enabled(train_encoder):
            image_features = encode_images(image_encoder, image_projection, images)
        logits = decoder(image_features, input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1),
            ignore_index=pad_id,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        n_samples += images.size(0)
    return total_loss / n_samples


def create_run(args, device) -> tuple:
    """Creates a time-stamped run directory below args.save_path and a log file listing all arguments.

    :return: (tuple) run directory, open log file
    """
    os.makedirs(args.save_path, exist_ok=True)
    run_directory = os.path.join(args.save_path, datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    os.makedirs(run_directory)
    log_file = open(os.path.join(run_directory, 'log.txt'), 'w')
    log_message(log_file, 'Arguments:')
    for name, value in vars(args).items():
        log_message(log_file, f'- {name}: {value}')
    log_message(log_file, f'- device: {device}')
    return run_directory, log_file


def save_checkpoint(run_directory, epoch, decoder, image_encoder, image_projection, optimizer, vocabulary, args) -> str:
    """Saves all model weights and the optimizer state. Returns the checkpoint path."""
    checkpoint_path = os.path.join(run_directory, f'captioner_epoch_{epoch:03d}.pt')
    torch.save({
        'epoch': epoch,
        'decoder': decoder.state_dict(),
        'image_encoder': image_encoder.state_dict(),
        'image_projection': None if image_projection is None else image_projection.state_dict(),
        'optimizer': optimizer.state_dict(),
        'vocabulary': vocabulary,
        'config': vars(args),
    }, checkpoint_path)
    return checkpoint_path


def log_wrong_predictions(log_file, wrong_examples, max_lines=None):
    """Logs the counts of (true caption, predicted caption) pairs, most frequent first (at most max_lines)."""
    log_message(log_file, 'Wrong held-out predictions (true -> predicted):')
    sorted_examples = sorted(wrong_examples.items(), key=lambda x: x[1], reverse=True)
    for (target, prediction), count in sorted_examples[:max_lines]:
        log_message(log_file, f'- {target} -> {prediction}: {count}')


def plot_examples(dataset, targets, predictions, n_examples, figure_name, seed=0) -> None:
    """Saves a figure with randomly chosen correct (top row) and wrong (bottom row) predictions.

    :param dataset: (Dataset) The data set the predictions were made on (not shuffled).
    :param targets: (list) True captions, in the same order as the data set.
    :param predictions: (list) Generated captions, in the same order as the data set.
    :param n_examples: (int) Maximum number of examples per row.
    :param figure_name: (str) File name of the figure.
    :param seed: (int) Random seed for choosing the examples.
    """
    correct = [index for index, (t, p) in enumerate(zip(targets, predictions)) if t == p]
    wrong = [index for index, (t, p) in enumerate(zip(targets, predictions)) if t != p]
    rng = random.Random(seed)
    rows = (
        rng.sample(correct, min(n_examples, len(correct))),
        rng.sample(wrong, min(n_examples, len(wrong))),
    )
    figure, axes = plt.subplots(2, n_examples, figsize=(3 * n_examples, 6.5), squeeze=False)
    for row_axes, indices in zip(axes, rows):
        for axis, index in zip(row_axes, indices):
            image, _ = dataset[index]
            axis.imshow(image.permute(1, 2, 0))
            axis.set_title(
                f'True: {targets[index]}\nPredicted: {predictions[index]}',
                color='green' if targets[index] == predictions[index] else 'red',
                fontsize=9,
            )
    for axis in axes.ravel():
        axis.axis('off')
    figure.suptitle(
        f'Held-out predictions: {len(correct)} correct (top), {len(wrong)} wrong (bottom)'
    )
    figure.tight_layout()
    figure.savefig(figure_name)
    plt.close(figure)
