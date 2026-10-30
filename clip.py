"""Small CLIP-style VLM experiment on colored geometric shapes."""
import argparse
import datetime
import itertools
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from my_utils import get_device, log_message
from torch.utils.data import DataLoader
from torchvision import transforms

import data.colored_shapes as cs
from loss.clip import contrastive_loss
from vlm_models import ImageEncoder, LinearProjectionHead, TextEncoder


def encode_texts(text_encoder, text_projection, captions, vocabulary, device) -> torch.Tensor:
    """Takes a list of captions and returns their encoded representations, based on the given encoders.

    :param text_encoder: (TextEncoder) The text encoder.
    :param text_projection: (LinearProjectionHead) The text projection head.
    :param captions: (list) The captions.
    :param vocabulary: (dict) The vocabulary.
    :param device: (torch.device) The device to use.
    :return: (torch.Tensor) The encoded texts.
    """
    token_ids = torch.stack(
        [cs.tokenize(caption, vocabulary) for caption in captions]
    ).to(device)
    return text_projection(text_encoder(token_ids))


def encode_images(image_encoder, image_projection, images, device) -> torch.Tensor:
    """Takes a tensor batch or list of images and returns their encoded representations.

    :param image_encoder: (ImageEncoder) The image encoder.
    :param image_projection: (LinearProjectionHead) The image projection head.
    :param images: (torch.Tensor | list) The images to encode.
    :param device: (torch.device) The device to use.
    :return: (torch.Tensor) The encoded images.
    """
    if isinstance(images, (list, tuple)):
        to_tensor = transforms.ToTensor()
        images = torch.stack([to_tensor(image) for image in images])
    return image_projection(image_encoder(images.to(device)))


def compute_similarity(image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity matrix between image and text embeddings.

    :param image_embeddings: (torch.Tensor) Image embeddings of shape (N, D).
    :param text_embeddings: (torch.Tensor) Text embeddings of shape (M, D).
    :return: (torch.Tensor) Cosine similarity matrix of shape (N, M).
    """
    image_embeddings = F.normalize(image_embeddings, dim=1)
    text_embeddings = F.normalize(text_embeddings, dim=1)
    return image_embeddings @ text_embeddings.T


def zero_shot_accuracy(
        image_encoder, image_projection, text_encoder, text_projection,
        loader, candidate_captions, vocabulary, device) -> float:
    """Compute zero-shot classification accuracy on a dataset.

    :param image_encoder: (ImageEncoder) The image encoder.
    :param image_projection: (LinearProjectionHead) The image projection head.
    :param text_encoder: (TextEncoder) The text encoder.
    :param text_projection: (LinearProjectionHead) The text projection head.
    :param loader: (DataLoader) The data loader.
    :param candidate_captions: (list) The candidate captions.
    :param vocabulary: (dict) The vocabulary.
    :param device: (torch.device) The device to use.
    :return: (float) The zero-shot accuracy.
    """
    image_encoder.eval()
    image_projection.eval()
    text_encoder.eval()
    text_projection.eval()
    with torch.no_grad():
        candidate_text_embeddings = encode_texts(
            text_encoder, text_projection, candidate_captions,
            vocabulary, device,
        )
        candidate_token_ids = torch.stack(
            [cs.tokenize(caption, vocabulary) for caption in candidate_captions]
        )
        correct = 0
        total = 0
        for images, token_ids in loader:
            image_embeddings = encode_images(
                image_encoder, image_projection, images, device
            )
            similarities = compute_similarity(image_embeddings, candidate_text_embeddings)
            predictions = similarities.argmax(dim=1)
            targets = torch.all(
                token_ids.unsqueeze(1) == candidate_token_ids.unsqueeze(0), dim=2
            )
            targets = targets.float().argmax(dim=1).to(device)
            correct += (predictions == targets).sum().item()
            total += token_ids.size(0)
    return correct / total


def load_checkpoint_models(checkpoint_path, args, device) -> tuple:
    """Load a CLIP checkpoint and return its models, vocabulary, and config.

    :param checkpoint_path: (str) The path to the checkpoint.
    :param args: (argparse.Namespace) The command-line arguments.
    :param device: (torch.device) The device to load the models onto.
    :return: (tuple) The models (as a tuple), vocabulary, and config.
    """
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    checkpoint_config = checkpoint.get('config', vars(args))
    model_args = argparse.Namespace(**checkpoint_config)
    vocabulary = checkpoint.get(
        'vocabulary',
        cs.make_vocabulary(getattr(model_args, 'include_position', False)),
    )
    models = create_models(model_args, vocabulary, device)
    state_names = (
        'image_encoder', 'text_encoder',
        'image_projection', 'text_projection',
    )
    for model, state_name in zip(models, state_names):
        model.load_state_dict(checkpoint[state_name])
        model.eval()
    return models, vocabulary, model_args


def make_dataset_grid(image_size, combinations, seed=0, position=None):
    """Plot one generated example for every color-shape combination.

    :param image_size: (int) The image size.
    :param combinations: (list) The color-shape combinations.
    :param seed: (int) The random seed.
    :param position: (tuple) The position.
    :return: (tuple) The figure and axes.
    """
    generator = cs.ShapeDataset(combinations, 1, image_size=image_size, seed=seed)
    # Create a figure with subfigures such that the color is on the x-axis and the shape is on the y-axis.
    figure, axes = plt.subplots(
        len(cs.SHAPES), len(cs.COLORS),
        figsize=(2.2 * len(cs.COLORS), 2.2 * len(cs.SHAPES)),
        squeeze=False,
    )

    # Every row represents a shape.
    for row, shape in enumerate(cs.SHAPES):
        # Every column represents a color.
        for column, color in enumerate(cs.COLORS):
            image = generator.make_image(
                color, shape, position=position,
                seed=seed + row * len(cs.COLORS) + column,
            )
            axes[row, column].imshow(image)
            axes[row, column].set_title(color)
            axes[row, column].axis('off')
        axes[row, 0].text(
            -0.18, 0.5, shape,
            transform=axes[row, 0].transAxes,
            ha='right', va='center', fontsize=12,
        )
    figure.suptitle('Generated shape dataset')
    figure.tight_layout()
    return figure


def get_test_visualization_samples(
        held_out_combinations, image_size, samples_per_combination, seed=0):
    """Return deterministic held-out images and their captions.

    :param held_out_combinations: (list) The held-out combinations.
    :param image_size: (int) The image size.
    :param samples_per_combination: (int) The number of samples per combination.
    :param seed: (int) The random seed.
    :return: (tuple) The images and captions, both as lists.
    """
    generator = cs.ShapeDataset(
        held_out_combinations, samples_per_combination,
        image_size=image_size, seed=seed,
    )
    images, captions = [], []
    for combination_index, combination in enumerate(held_out_combinations):
        for sample_index in range(samples_per_combination):
            image_seed = seed + combination_index * samples_per_combination + sample_index
            images.append(generator.make_image(*combination, seed=image_seed))
            captions.append(cs.combination_caption(combination))
    return images, captions


def predict_captions(
        images: list, image_encoder, image_projection, text_encoder,
        text_projection, captions: tuple | list, vocabulary: dict, device: torch.device) -> tuple:
    """Return the best caption, similarity, and full score matrix per image.

    :param images: (list) A list of images.
    :param image_encoder: The image encoder.
    :param image_projection: The image projection.
    :param text_encoder: The text encoder.
    :param text_projection: The text projection.
    :param captions: (tuple | list) A list of captions.
    :param vocabulary: The vocabulary.
    :param device: The device.
    :return: (tuple) Predicted captions, best scores, and similarity matrix.
    """
    image_encoder.eval()
    image_projection.eval()
    text_encoder.eval()
    text_projection.eval()
    with torch.no_grad():
        image_embeddings = encode_images(
            image_encoder, image_projection, images, device
        )
        text_embeddings = encode_texts(
            text_encoder, text_projection, captions, vocabulary, device
        )
        similarities = compute_similarity(image_embeddings, text_embeddings)
    # Look for the best caption for each image by finding the index of the maximum similarity score
    best_indices = similarities.argmax(dim=1).cpu().tolist()
    best_scores = similarities.max(dim=1).values.cpu().tolist()
    return (
        [captions[index] for index in best_indices],
        best_scores,
        similarities.cpu(),
    )


def plot_prediction_grid(
        images, true_captions, predicted_captions, scores, figure_name) -> plt.Figure:
    """Plot held-out images with their zero-shot predictions.

    :param images: List of images to plot.
    :param true_captions: List of true captions for the images.
    :param predicted_captions: List of predicted captions for the images.
    :param scores: List of scores for the predicted captions.
    :param figure_name: Name of the file to save the figure as.
    """
    n_columns = 2
    n_rows = (len(images) + n_columns - 1) // n_columns
    figure, axes = plt.subplots(
        n_rows, n_columns,
        figsize=(12, max(3, 4 * n_rows)),
        squeeze=False,
    )
    axes = axes.ravel()
    # Show in each subplot one image, its true caption and the predicted caption plus the score.
    for axis, image, true_caption, predicted_caption, score in zip(
            axes, images, true_captions, predicted_captions, scores):
        axis.imshow(image)
        axis.axis('off')
        status = 'correct' if true_caption == predicted_caption else 'wrong'
        axis.set_title(
            f'True: {true_caption} \n Predicted: {predicted_caption} \n'
            f'({score:.3f}, {status})',
            color='green' if status == 'correct' else 'red',
        )
    for axis in axes[len(images):]:
        axis.axis('off')
    figure.suptitle('Held-out zero-shot predictions')
    figure.tight_layout()
    figure.savefig(figure_name)
    return figure


def plot_similarity_heatmap(
        similarities, true_captions, candidate_captions, figure_name):
    """Plot image-to-caption similarities for held-out examples.

    Creates a heatmap of cosine similarities between images and candidate captions.
    The images are in the vertical axis and the captions are in the horizontal axis.
    So, each row represents an image and each column represents a candidate caption.

    :param similarities: Tensor of shape (num_images, num_candidate_captions)
        containing cosine similarities between images and candidate captions.
    :param true_captions: List of true captions for the held-out images.
    :param candidate_captions: List of candidate captions to compare against.
    :param figure_name: Name of the file to save the figure as.
    """
    figure, axis = plt.subplots(
        figsize=(max(14, 0.65 * len(candidate_captions)),
                 max(5, 0.45 * len(true_captions))),
    )
    heatmap = axis.imshow(similarities.numpy(), aspect='auto', cmap='viridis')
    axis.set_xticks(range(len(candidate_captions)))
    axis.set_xticklabels(candidate_captions, rotation=60, ha='right')
    axis.set_yticks(range(len(true_captions)))
    axis.set_yticklabels(true_captions)
    axis.set_xlabel('Candidate caption')
    axis.set_ylabel('Test image true caption')
    axis.set_title('Image-caption cosine similarities')
    figure.colorbar(heatmap, ax=axis, label='Cosine similarity')
    figure.tight_layout()
    figure.savefig(figure_name)
    return figure


def evaluate_checkpoint(checkpoint_path, args, tasks):
    """Run selected visual evaluations for one saved CLIP checkpoint.

    :param checkpoint_path: Path to the checkpoint to evaluate.
    :param args: Command-line arguments.
    :param tasks: List of tasks to evaluate.
    """
    device = get_device(args)
    (image_encoder, text_encoder, image_projection, text_projection), vocabulary, config = (
        load_checkpoint_models(checkpoint_path, args, device)
    )
    output_directory = os.path.dirname(checkpoint_path)
    all_combinations, _, held_out_combinations = cs.get_experiment_combinations(
        getattr(config, 'include_position', False)
    )
    all_captions = cs.candidate_captions(all_combinations)

    if 'dataset-grid' in tasks:
        grid_combinations = tuple(itertools.product(cs.COLORS, cs.SHAPES))
        figure = make_dataset_grid(
            config.image_size,
            grid_combinations,
            seed=config.seed,
            position='center' if getattr(config, 'include_position', False) else None,
        )
        figure.savefig(os.path.join(output_directory, 'dataset_grid.png'))
        plt.close(figure)

    if 'predictions' in tasks or 'heatmap' in tasks:
        images, true_captions = get_test_visualization_samples(
            held_out_combinations,
            config.image_size,
            args.visualization_samples,
            seed=config.seed + 100000,
        )
        predicted_captions, scores, similarities = predict_captions(
            images, image_encoder, image_projection,
            text_encoder, text_projection, all_captions,
            vocabulary, device,
        )

        if 'predictions' in tasks:
            figure_name = os.path.join(output_directory, 'zero_shot_predictions.png')
            figure = plot_prediction_grid(
                images, true_captions, predicted_captions, scores, figure_name
            )
            plt.close(figure)

        if 'heatmap' in tasks:
            figure_name = os.path.join(output_directory, 'similarity_heatmap.png')
            figure = plot_similarity_heatmap(
                similarities, true_captions, all_captions, figure_name
            )
            plt.close(figure)


def create_models(args, vocabulary, device) -> tuple:
    """Create the image encoder, text encoder, and both projection heads.

    :param args: (argparse.Namespace) command-line arguments
    :param vocabulary: (dict) with word-to-index mapping
    :param device: (torch.device) device on which to create the models
    :return: (tuple) image encoder, text encoder, image projection head, text projection head
    """
    image_encoder = ImageEncoder(
        img_channels=3, img_size=args.image_size, latent_dims=args.image_dim,
    ).to(device)

    text_encoder = TextEncoder(
        len(vocabulary), embedding_dim=args.text_dim,
    ).to(device)

    image_projection = LinearProjectionHead(
        args.image_dim, output_dim=args.shared_dim,
    ).to(device)

    text_projection = LinearProjectionHead(
        args.text_dim, output_dim=args.shared_dim,
    ).to(device)

    return image_encoder, text_encoder, image_projection, text_projection


def save_checkpoint(
        run_directory, epoch, image_encoder, text_encoder,
        image_projection, text_projection, optimizer, vocabulary, args) -> str:
    """Save all model weights and optimizer state

    :param run_directory: (str) directory where to save the checkpoint
    :param epoch: (int) epoch number
    :param image_encoder: (nn.Module) image encoder (we save the state dict from it)
    :param text_encoder: (nn.Module) text encoder
    :param image_projection: (nn.Module) image projection head
    :param text_projection: (nn.Module) text projection head
    :param optimizer: (torch.optim.Optimizer) optimizer
    :param vocabulary: (dict) vocabulary
    :param args: (argparse.Namespace) command-line arguments
    :return: (str): checkpoint path
    ."""

    checkpoint_path = os.path.join(run_directory, f'clip_epoch_{epoch:03d}.pt')
    torch.save({
        'epoch': epoch,
        'image_encoder': image_encoder.state_dict(),
        'text_encoder': text_encoder.state_dict(),
        'image_projection': image_projection.state_dict(),
        'text_projection': text_projection.state_dict(),
        'optimizer': optimizer.state_dict(),
        'vocabulary': vocabulary,
        'config': vars(args),
    }, checkpoint_path)
    return checkpoint_path


def main(args):
    """Main function to train text and vision encoder plus projection heads as proposed by CLIP.

    :param args: Command-line arguments.
    :return: None
    """

    # Do some preparation
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args)
    # Make a dict with index as key and word as value with all words.
    vocabulary = cs.make_vocabulary(args.include_position)
    all_combinations, train_combinations, held_out_combinations = cs.get_experiment_combinations(
        args.include_position
    )

    # Create out directory
    os.makedirs(args.save_path, exist_ok=True)
    run_name = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    run_directory = os.path.join(args.save_path, run_name)
    os.makedirs(run_directory)
    # Create log file and add parameters to it
    log_path = os.path.join(run_directory, 'log.txt')
    log_file = open(log_path, 'w')
    log_message(log_file, 'Arguments:')
    for name, value in vars(args).items():
        log_message(log_file, f'- {name}: {value}')
    log_message(log_file, f'- device: {device}')

    # Get training and test datasets and wrap them in dataloaders
    train_dataset = cs.ShapeDataset(
        train_combinations, args.samples_per_combination,
        image_size=args.image_size, seed=args.seed,
    )
    test_dataset = cs.ShapeDataset(
        held_out_combinations, args.test_samples_per_combination,
        image_size=args.image_size, seed=args.seed + 100000,
    )
    collate_fn = lambda batch: cs.collate_shape_batch(batch, vocabulary)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn,
    )

    # Get the four models we train (or evaluate)
    image_encoder, text_encoder, image_projection, text_projection = create_models(
        args, vocabulary, device
    )
    # Optimizer...
    optimizer = torch.optim.Adam(
        itertools.chain(
            image_encoder.parameters(), text_encoder.parameters(),
            image_projection.parameters(), text_projection.parameters(),
        ),
        lr=args.lr,
    )

    # Now, we're ready for model the training
    for epoch in range(args.epochs):
        image_encoder.train()
        text_encoder.train()
        image_projection.train()
        text_projection.train()
        total_loss = 0.0

        # For each batch of the training data do a training step (should this go in a separate function?)
        for images, token_ids in train_loader:
            images = images.to(device)
            token_ids = token_ids.to(device)
            image_embeddings = image_projection(image_encoder(images))
            text_embeddings = text_projection(text_encoder(token_ids))
            loss = contrastive_loss(
                image_embeddings, text_embeddings, token_ids, args.temperature
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        # After one epoch of training we evaluate the model on the held-out test set
        test_accuracy = zero_shot_accuracy(
            image_encoder, image_projection, text_encoder, text_projection,
            test_loader, cs.candidate_captions(all_combinations),
            vocabulary, device,
        )
        # Output loss and training accuraccy. And log it
        mean_loss = total_loss / len(train_dataset)
        message = (
            f'Epoch {epoch + 1:3d} | Loss: {mean_loss:.4f} | '
            f'Held-out zero-shot accuracy: {test_accuracy:.2%}'
        )
        log_message(log_file, message)
        # Save the models
        last_checkpoint_path = save_checkpoint(
            run_directory,
            epoch + 1,
            image_encoder,
            text_encoder,
            image_projection,
            text_projection,
            optimizer,
            vocabulary,
            args,
        )

    log_message(log_file, 'Training combinations: ' + ', '.join(
        cs.combination_caption(combination) for combination in train_combinations
    ))
    log_message(log_file, 'Held-out combinations: ' + ', '.join(
        cs.combination_caption(combination) for combination in held_out_combinations
    ))
    log_file.close()
    return last_checkpoint_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a small CLIP on colored geometric shapes'
    )
    parser.add_argument(
        '--operation', choices=('train', 'evaluate', 'both'), default='train',
        help='train, evaluate a checkpoint, or do both (default: train)',
    )
    parser.add_argument(
        '--checkpoint-path',
        help='checkpoint to evaluate; with both, the newly trained checkpoint is used',
    )
    parser.add_argument(
        '--evaluation-tasks',
        nargs='+',
        choices=('dataset-grid', 'predictions', 'heatmap'),
        default=('dataset-grid', 'predictions', 'heatmap'),
        help='visualizations to create during evaluation',
    )
    parser.add_argument(
        '--visualization-samples', type=int, default=2,
        help='held-out examples per combination in prediction/heatmap plots',
    )
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--image-dim', type=int, default=64, help='image dimension')
    parser.add_argument('--text-dim', type=int, default=64, help='text embedding dimension')
    parser.add_argument('--shared-dim', type=int, default=32, help='shared embedding dimension')
    parser.add_argument('--samples-per-combination', type=int, default=500)
    parser.add_argument('--test-samples-per-combination', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--include-position', action='store_true',
        help='include position in captions and images (default: color and shape only)',
    )
    parser.add_argument('--save-path', type=str, default='out_clip')
    parser.add_argument('--no-gpu', action='store_true')
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    if args.operation in ('train', 'both'):
        checkpoint_path = main(args)
    if args.operation in ('evaluate', 'both'):
        if checkpoint_path is None:
            parser.error('--checkpoint-path is required for evaluation')
        evaluate_checkpoint(checkpoint_path, args, args.evaluation_tasks)
