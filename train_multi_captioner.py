"""Train a caption decoder for images with two shapes, e.g. "red circle left of green square".

The image encoder (SpatialImageEncoder) returns a grid of visual tokens instead of one vector, so the decoder
can attend to the left object while describing it and to the right object afterwards.
Its first three conv layers can be loaded from a CLIP checkpoint (see clip.py) and are then frozen by default.
"""
import argparse
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from my_utils import get_device, log_message
from torch.utils.data import DataLoader

import data.multi_shapes as ms
from data.colored_shapes import collate_shape_batch
from captioning import create_run, log_wrong_predictions, plot_examples, predict_captions, save_checkpoint, train_epoch
from clip import load_checkpoint_models
from vlm_models import CaptionDecoder, SpatialImageEncoder


def evaluate(image_encoder, decoder, loader, vocabulary, inverse_vocabulary, held_out_objects, device) -> dict:
    """Generates captions for a data set and compares them to the true captions.

    :param held_out_objects: (tuple) Objects never seen in training. Used to compare the accuracy
        on held-out objects with the accuracy on the other objects.
    :return: (dict) metrics (fractions of all samples or objects), counts of wrong predictions,
        and all true and predicted captions (in the order of the loader).
    """
    targets, predictions = predict_captions(
        image_encoder, None, decoder, loader, vocabulary, inverse_vocabulary, device,
    )
    counts = defaultdict(lambda: 0)
    examples = defaultdict(lambda: 0)
    for target, prediction in zip(targets, predictions):
        target_objects = ms.parse_caption(target)
        predicted_objects = ms.parse_caption(prediction)
        counts['exact'] += prediction == target
        if prediction != target:
            examples[(target, prediction)] += 1
        has_held_out_object = any(obj in held_out_objects for obj in target_objects)
        counts['samples with held-out object'] += has_held_out_object
        if predicted_objects is None:
            counts['malformed'] += 1
            continue

        for side, target_object, predicted_object in zip(('left', 'right'), target_objects, predicted_objects):
            counts[f'{side} object'] += predicted_object == target_object
            counts['color'] += predicted_object[0] == target_object[0]
            counts['shape'] += predicted_object[1] == target_object[1]
            kind = 'held-out object' if target_object in held_out_objects else 'other object'
            counts[f'n {kind}'] += 1
            counts[kind] += predicted_object == target_object

        if prediction != target:
            target_colors, target_shapes = zip(*target_objects)
            predicted_colors, predicted_shapes = zip(*predicted_objects)
            if predicted_objects == target_objects[::-1]:
                # Right objects, but on the wrong sides
                counts['order swapped'] += 1
            elif sorted(predicted_colors) == sorted(target_colors) and sorted(predicted_shapes) == sorted(target_shapes):
                # Right colors and shapes, but combined wrongly, e.g. "green circle left of red square"
                counts['binding errors'] += 1
        if has_held_out_object and not any(obj in held_out_objects for obj in predicted_objects):
            counts['only training objects'] += 1

    n_samples = len(targets)
    metrics = {
        'exact': counts['exact'] / n_samples,
        'left object': counts['left object'] / n_samples,
        'right object': counts['right object'] / n_samples,
        'color': counts['color'] / (2 * n_samples),
        'shape': counts['shape'] / (2 * n_samples),
    }
    for kind in ('held-out object', 'other object'):
        if counts[f'n {kind}']:
            metrics[kind] = counts[kind] / counts[f'n {kind}']
    if counts['samples with held-out object']:
        metrics['only training objects'] = counts['only training objects'] / counts['samples with held-out object']
    for error in ('order swapped', 'binding errors', 'malformed'):
        metrics[error] = counts[error] / n_samples
    return {
        'metrics': metrics,
        'wrong_examples': examples,
        'targets': targets,
        'predictions': predictions,
    }


def format_results(name, results) -> str:
    return f'{name}: ' + ', '.join(
        f'{metric} {value:.1%}' for metric, value in results['metrics'].items()
    )


def plot_dataset_examples(train_dataset, test_dataset, n_examples, figure_name):
    """Saves a figure with training images (top row) and held-out test images (bottom row) and their captions."""
    figure, axes = plt.subplots(2, n_examples, figsize=(3 * n_examples, 6.5), squeeze=False)
    for row_axes, dataset, name in zip(axes, (train_dataset, test_dataset), ('Train', 'Held-out')):
        for index, axis in enumerate(row_axes):
            image, caption = dataset[index]
            axis.imshow(image.permute(1, 2, 0))
            axis.set_title(f'{name}:\n{caption}', fontsize=9)
            axis.axis('off')
    figure.tight_layout()
    figure.savefig(figure_name)
    plt.close(figure)


def create_image_encoder(args, device, log_file) -> SpatialImageEncoder:
    """Creates the spatial image encoder and loads conv0-2 from a CLIP checkpoint if one is given."""
    image_encoder = SpatialImageEncoder(
        img_channels=3, out_channels=args.encoder_channels, grid_size=args.grid_size,
    ).to(device)
    if args.checkpoint_path is not None:
        (clip_image_encoder, _, _, _), _, clip_config = load_checkpoint_models(args.checkpoint_path, args, device)
        image_encoder.load_image_encoder_weights(clip_image_encoder.state_dict())
        log_message(log_file, f'Loaded conv0-2 from CLIP checkpoint (trained on {clip_config.image_size} px images)')
        if not args.train_pretrained_layers:
            image_encoder.freeze_pretrained_layers()
    return image_encoder


def main(args):
    """Train the spatial image encoder and the caption decoder and evaluate them on held-out objects."""
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args)
    run_directory, log_file = create_run(args, device)

    vocabulary = ms.make_vocabulary()
    inverse_vocabulary = {index: word for word, index in vocabulary.items()}
    _, train_objects, held_out_objects = ms.get_objects()

    # The decoder input starts with <bos>, the target ends with <eos>
    collate_fn = lambda batch: collate_shape_batch(
        [(image, f'<bos> {caption} <eos>') for image, caption in batch], vocabulary
    )
    train_dataset = ms.MultiShapeDataset(
        train_objects, args.train_samples, image_size=args.image_size, seed=args.seed,
    )
    # Fresh images with training objects only, to compare against images with a held-out object
    train_eval_dataset = ms.MultiShapeDataset(
        train_objects, args.test_samples, image_size=args.image_size, seed=args.seed + 200000,
    )
    # Every test image contains one held-out object and one training object
    test_dataset = ms.MultiShapeDataset(
        train_objects, args.test_samples, image_size=args.image_size, seed=args.seed + 100000,
        required_objects=held_out_objects,
    )
    plot_dataset_examples(
        train_dataset, test_dataset, args.visual_examples, os.path.join(run_directory, 'dataset_examples.png')
    )
    if args.plot_dataset_only:
        log_file.close()
        return
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    image_encoder = create_image_encoder(args, device, log_file)
    decoder = CaptionDecoder(
        len(vocabulary), image_encoder.latent_dim, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, max_length=ms.MAX_CAPTION_WORDS + 1, dropout=args.dropout,
        padding_idx=vocabulary['<pad>'], n_image_tokens=image_encoder.n_tokens,
    ).to(device)
    # Frozen pretrained layers have requires_grad == False
    parameters = [p for p in (*image_encoder.parameters(), *decoder.parameters()) if p.requires_grad]
    log_message(log_file, f'Visual tokens per image: {image_encoder.n_tokens}')
    log_message(log_file, f'Trainable parameters: {sum(p.numel() for p in parameters):,}')
    optimizer = torch.optim.Adam(parameters, lr=args.lr)

    for epoch in range(args.epochs):
        # The encoder is always trained (at least conv3); frozen layers stay in eval mode
        mean_loss = train_epoch(
            image_encoder, None, decoder, train_loader, optimizer, vocabulary['<pad>'], True, device,
        )
        train_results = evaluate(
            image_encoder, decoder, train_eval_loader, vocabulary, inverse_vocabulary, held_out_objects, device,
        )
        test_results = evaluate(
            image_encoder, decoder, test_loader, vocabulary, inverse_vocabulary, held_out_objects, device,
        )
        log_message(log_file, f'Epoch {epoch + 1:3d} | Loss: {mean_loss:.4f}')
        log_message(log_file, '    ' + format_results('Train objects  ', train_results))
        log_message(log_file, '    ' + format_results('Held-out object', test_results))
        save_checkpoint(run_directory, epoch + 1, decoder, image_encoder, None, optimizer, vocabulary, args)

    log_wrong_predictions(log_file, test_results['wrong_examples'], max_lines=20)
    log_message(log_file, 'Held-out objects: ' + ', '.join(f'{color} {shape}' for color, shape in held_out_objects))
    plot_examples(
        test_dataset, test_results['targets'], test_results['predictions'],
        args.visual_examples, os.path.join(run_directory, 'held_out_examples.png'), seed=args.seed,
    )
    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a caption decoder for images with two colored shapes'
    )
    parser.add_argument(
        '--checkpoint-path',
        help='CLIP checkpoint (from clip.py) whose conv0-2 weights are used; '
             'if omitted, the whole image encoder is trained from scratch',
    )
    parser.add_argument(
        '--train-pretrained-layers', action='store_true',
        help='fine-tune the conv layers loaded from the checkpoint instead of keeping them frozen',
    )
    parser.add_argument('--grid-size', type=int, default=4, help='the encoder returns grid-size x grid-size tokens')
    parser.add_argument('--encoder-channels', type=int, default=32, help='dimension of each visual token')
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--train-samples', type=int, default=20000)
    parser.add_argument('--test-samples', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--visual-examples', type=int, default=4,
        help='number of examples per row in dataset_examples.png and held_out_examples.png',
    )
    parser.add_argument(
        '--plot-dataset-only', action='store_true',
        help='only save dataset_examples.png to check the generated images, no training',
    )
    parser.add_argument('--save-path', type=str, default='out_multi_captioner')
    parser.add_argument('--no-gpu', action='store_true')
    main(parser.parse_args())
