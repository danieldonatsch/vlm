"""Train a small transformer decoder that generates captions for the colored shapes images.

The image encoder either comes from a trained CLIP checkpoint (see clip.py) and is frozen by default,
or it is trained from scratch together with the decoder when no checkpoint is given.
"""
import argparse
import datetime
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from my_utils import get_device, log_message
from torch.utils.data import DataLoader

import data.colored_shapes as cs
from clip import load_checkpoint_models
from vlm_models import CaptionDecoder, ImageEncoder


# Order of the words in the generated captions. The data set produces "color shape [position]".
CAPTION_ORDERS = {
    'color-first': ('color', 'shape', 'position'),
    'shape-first': ('shape', 'color', 'position'),
    'position-first': ('position', 'color', 'shape'),
}


def reorder_caption(caption, caption_order) -> str:
    """Reorders a "color shape [position]" caption, e.g. to "shape color [position]" for shape-first.

    :param caption: (str) Caption as produced by the data set.
    :param caption_order: (str) One of the keys of CAPTION_ORDERS.
    :return: (str) Reordered caption.
    """
    words = dict(zip(('color', 'shape', 'position'), caption.split()))
    return ' '.join(words[slot] for slot in CAPTION_ORDERS[caption_order] if slot in words)


def encode_images(image_encoder, image_projection, images) -> torch.Tensor:
    """Returns the image features which are fed to the decoder.

    :param image_encoder: (ImageEncoder) The image encoder.
    :param image_projection: (LinearProjectionHead | None) CLIP projection head; if None, the raw encoder output is used.
    :param images: (torch.Tensor) Batch of images.
    :return: (torch.Tensor) Image features of shape (B, D).
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


def evaluate(image_encoder, image_projection, decoder, loader, vocabulary, inverse_vocabulary,
             train_captions, device) -> dict:
    """Generates captions for a data set and compares them to the true captions.

    :param train_captions: (set) Captions seen during training. Used to measure how often the model
        falls back to a known combination instead of producing a new one.
    :return: (dict) exact-match accuracy, accuracy per word slot (color, shape, position),
        fraction of predictions which are a training caption, counts of wrong predictions,
        and all true and predicted captions (in the order of the loader).
    """
    image_encoder.eval()
    decoder.eval()
    if image_projection is not None:
        image_projection.eval()
    n_correct, n_total, n_seen = 0, 0, 0
    slot_correct = None
    examples = defaultdict(lambda: 0)
    all_targets, all_predictions = [], []
    with torch.no_grad():
        for images, token_ids in loader:
            predictions = generate_captions(
                image_encoder, image_projection, decoder, images,
                vocabulary, inverse_vocabulary, device,
            )
            # Strip <bos>/<eos> from the true captions
            targets = decode_tokens(
                token_ids[:, 1:], inverse_vocabulary, vocabulary['<eos>'], vocabulary['<pad>']
            )
            for prediction, target in zip(predictions, targets):
                target_words = target.split()
                prediction_words = prediction.split()
                if slot_correct is None:
                    slot_correct = [0] * len(target_words)
                for slot, target_word in enumerate(target_words):
                    if slot < len(prediction_words) and prediction_words[slot] == target_word:
                        slot_correct[slot] += 1
                n_correct += prediction == target
                n_seen += prediction in train_captions
                n_total += 1
                if prediction != target:
                    examples[(target, prediction)] += 1
            all_targets.extend(targets)
            all_predictions.extend(predictions)
    return {
        'accuracy': n_correct / n_total,
        'slot_accuracy': [count / n_total for count in slot_correct],
        'seen_fraction': n_seen / n_total,
        'wrong_examples': examples,
        'targets': all_targets,
        'predictions': all_predictions,
    }


def plot_examples(dataset, targets, predictions, n_examples, figure_name, seed=0) -> None:
    """Saves a figure with randomly chosen correct (top row) and wrong (bottom row) predictions.

    :param dataset: (ShapeDataset) The data set the predictions were made on (not shuffled).
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


def format_results(name, results, caption_order) -> str:
    # zip() stops after the last word, so 'position' is dropped for captions without position
    slots = ', '.join(
        f'{slot_name} {accuracy:.1%}'
        for slot_name, accuracy in zip(CAPTION_ORDERS[caption_order], results['slot_accuracy'])
    )
    return (
        f'{name}: exact {results["accuracy"]:.2%} ({slots}) | '
        f'predicted a training caption: {results["seen_fraction"]:.1%}'
    )


def create_image_encoder(args, device) -> tuple:
    """Loads the image encoder (and projection) from a CLIP checkpoint or creates a new one.

    :return: (tuple) image encoder, image projection (or None), include_position, image size
    """
    if args.checkpoint_path is None:
        image_encoder = ImageEncoder(
            img_channels=3, img_size=args.image_size, latent_dims=args.image_dim,
        ).to(device)
        return image_encoder, None, args.include_position, args.image_size

    (image_encoder, _, image_projection, _), _, clip_config = load_checkpoint_models(
        args.checkpoint_path, args, device
    )
    if args.image_features == 'encoder':
        image_projection = None
    if not args.train_encoder:
        for module in (image_encoder, image_projection):
            if module is not None:
                module.requires_grad_(False)
    return (
        image_encoder, image_projection,
        getattr(clip_config, 'include_position', False), clip_config.image_size,
    )


def main(args):
    """Train the caption decoder (and optionally the image encoder) and evaluate it on held-out combinations."""
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args)

    image_encoder, image_projection, include_position, image_size = create_image_encoder(args, device)
    encoder_is_trained = args.checkpoint_path is None or args.train_encoder
    vocabulary = cs.make_caption_vocabulary(include_position)
    inverse_vocabulary = {index: word for word, index in vocabulary.items()}
    _, train_combinations, held_out_combinations = cs.get_experiment_combinations(include_position)
    if args.caption_order == 'position-first' and not include_position:
        raise ValueError('--caption-order position-first requires captions with positions')
    train_captions = {
        reorder_caption(caption, args.caption_order)
        for caption in cs.candidate_captions(train_combinations)
    }

    # Create out directory and log file
    os.makedirs(args.save_path, exist_ok=True)
    run_directory = os.path.join(args.save_path, datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    os.makedirs(run_directory)
    log_file = open(os.path.join(run_directory, 'log.txt'), 'w')
    log_message(log_file, 'Arguments:')
    for name, value in vars(args).items():
        log_message(log_file, f'- {name}: {value}')
    log_message(log_file, f'- device: {device}')
    log_message(log_file, f'- include_position (from checkpoint if given): {include_position}')

    # The decoder input starts with <bos>, the target ends with <eos>
    collate_fn = lambda batch: cs.collate_shape_batch(
        [(image, f'<bos> {reorder_caption(caption, args.caption_order)} <eos>') for image, caption in batch],
        vocabulary,
    )
    train_dataset = cs.ShapeDataset(
        train_combinations, args.samples_per_combination, image_size=image_size, seed=args.seed,
    )
    # Fresh images of the training combinations, to compare against the held-out combinations
    train_eval_dataset = cs.ShapeDataset(
        train_combinations, args.test_samples_per_combination, image_size=image_size, seed=args.seed + 200000,
    )
    test_dataset = cs.ShapeDataset(
        held_out_combinations, args.test_samples_per_combination, image_size=image_size, seed=args.seed + 100000,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    with torch.no_grad():
        image_dim = encode_images(
            image_encoder.eval(), image_projection, torch.zeros(1, 3, image_size, image_size, device=device)
        ).size(1)
    # Longest caption has 3 words; plus <bos> as decoder input
    decoder = CaptionDecoder(
        len(vocabulary), image_dim, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, max_length=4, dropout=args.dropout,
        padding_idx=vocabulary['<pad>'],
    ).to(device)
    log_message(log_file, f'Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}')

    parameters = list(decoder.parameters())
    if encoder_is_trained:
        parameters += list(image_encoder.parameters())
        if image_projection is not None:
            parameters += list(image_projection.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr)

    for epoch in range(args.epochs):
        decoder.train()
        # A frozen encoder stays in eval mode so that its BatchNorm statistics are not changed
        image_encoder.train(encoder_is_trained)
        if image_projection is not None:
            image_projection.train(encoder_is_trained)
        total_loss = 0.0

        for images, token_ids in train_loader:
            images = images.to(device)
            token_ids = token_ids.to(device)
            # Teacher forcing: input is [<bos>, w1, w2], target is [w1, w2, <eos>]
            input_ids = token_ids[:, :-1]
            target_ids = token_ids[:, 1:]
            with torch.set_grad_enabled(encoder_is_trained):
                image_features = encode_images(image_encoder, image_projection, images)
            logits = decoder(image_features, input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1),
                ignore_index=vocabulary['<pad>'],
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        train_results = evaluate(
            image_encoder, image_projection, decoder, train_eval_loader,
            vocabulary, inverse_vocabulary, train_captions, device,
        )
        test_results = evaluate(
            image_encoder, image_projection, decoder, test_loader,
            vocabulary, inverse_vocabulary, train_captions, device,
        )
        log_message(log_file, f'Epoch {epoch + 1:3d} | Loss: {total_loss / len(train_dataset):.4f}')
        log_message(log_file, '    ' + format_results('Train combinations   ', train_results, args.caption_order))
        log_message(log_file, '    ' + format_results('Held-out combinations', test_results, args.caption_order))

        checkpoint_path = os.path.join(run_directory, f'captioner_epoch_{epoch + 1:03d}.pt')
        torch.save({
            'epoch': epoch + 1,
            'decoder': decoder.state_dict(),
            'image_encoder': image_encoder.state_dict(),
            'image_projection': None if image_projection is None else image_projection.state_dict(),
            'optimizer': optimizer.state_dict(),
            'vocabulary': vocabulary,
            'config': vars(args),
        }, checkpoint_path)

    log_message(log_file, 'Wrong held-out predictions (true -> predicted):')
    for (target, prediction), count in sorted(test_results['wrong_examples'].items(), key=lambda x: x[1], reverse=True):
        log_message(log_file, f'- {target} -> {prediction}: {count}')
    log_message(log_file, 'Held-out combinations: ' + ', '.join(
        cs.combination_caption(combination) for combination in held_out_combinations
    ))
    plot_examples(
        test_dataset, test_results['targets'], test_results['predictions'],
        args.visual_examples, os.path.join(run_directory, 'held_out_examples.png'), seed=args.seed,
    )
    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a small transformer decoder to caption colored geometric shapes'
    )
    parser.add_argument(
        '--checkpoint-path',
        default='out_clip/3w_lr=5e-4_temp=02/vlm_epoch_020.pt',
        help='CLIP checkpoint (from clip.py) whose image encoder is used; '
             'if omitted, a new image encoder is trained together with the decoder',
    )
    parser.add_argument(
        '--image-features', choices=('projection', 'encoder'), default='projection',
        help='feed the decoder the CLIP embedding (projection) or the raw encoder output (encoder)',
    )
    parser.add_argument(
        '--train-encoder', action='store_true',
        help='fine-tune the CLIP image encoder instead of keeping it frozen',
    )
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    # Only used when no checkpoint is given; otherwise taken from the CLIP checkpoint
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--image-dim', type=int, default=64)
    parser.add_argument('--text-dim', type=int, default=64, help='text embedding dimension')
    parser.add_argument('--shared-dim', type=int, default=32, help='shared embedding dimension')
    parser.add_argument(
        '--include-position', action='store_true',
        help='include position in captions and images (ignored if a checkpoint is given)',
    )
    parser.add_argument('--samples-per-combination', type=int, default=500)
    parser.add_argument('--test-samples-per-combination', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument(
        '--caption-order', choices=tuple(CAPTION_ORDERS), default='color-first',
        help='word order of the generated captions; position-first requires positions',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--visual-examples', type=int, default=4,
        help='number of correct and of wrong held-out predictions shown in held_out_examples.png',
    )
    parser.add_argument('--save-path', type=str, default='out_captioner')
    parser.add_argument('--no-gpu', action='store_true')
    main(parser.parse_args())
