"""Show where the caption decoder of train_multi_captioner.py looks while it generates each word.

For every generated word, the attention from the decoder position that predicts this word to the
grid of visual tokens is drawn as a heatmap over the image.
"""
import argparse
import os
import random

import matplotlib.pyplot as plt
import torch

from my_utils import get_device

import data.multi_shapes as ms
from captioning import decode_tokens
from train_multi_captioner import create_datasets, load_multi_captioner


def word_attention(image_encoder, decoder, image, vocabulary, inverse_vocabulary, layer, head) -> tuple:
    """Generates a caption for one image and returns the attention of every word to the visual tokens.

    :param layer: (int) index of the decoder layer (0-based)
    :param head: (int | None) attention head; if None, the mean over all heads is used
    :return: (tuple) generated words, attention maps of shape (n_words, grid_size, grid_size),
        share of the attention that goes to the visual tokens (instead of the caption tokens) per word
    """
    image_features = image_encoder(image.unsqueeze(0))
    token_ids = decoder.generate(image_features, vocabulary['<bos>'], vocabulary['<eos>'])
    words = decode_tokens(token_ids, inverse_vocabulary, vocabulary['<eos>'], vocabulary['<pad>'])[0].split()
    # The decoder input is <bos> followed by the generated words; position i predicts word i
    input_ids = torch.tensor(
        [[vocabulary['<bos>']] + [vocabulary[word] for word in words[:-1]]], device=image.device
    )
    weights = decoder.attention_maps(image_features, input_ids)[layer][0]
    weights = weights.mean(dim=0) if head is None else weights[head]
    n_image_tokens = decoder.n_image_tokens
    # Rows: positions of <bos>, word 1, ...; columns: the visual tokens
    image_attention = weights[n_image_tokens:, :n_image_tokens]
    grid_size = int(n_image_tokens ** 0.5)
    return (
        words,
        image_attention.reshape(len(words), grid_size, grid_size).cpu(),
        image_attention.sum(dim=1).cpu(),
    )


def plot_attention(images, true_captions, results, grid_extent, figure_name, title):
    """Saves a figure with one row per image: the image itself and one attention heatmap per generated word.

    :param grid_extent: (tuple) pixel range (start, end) covered by the token grid, see token_grid_extent()
    """
    start, end = grid_extent
    n_columns = 1 + max(len(words) for words, _, _ in results)
    figure, axes = plt.subplots(
        len(images), n_columns, figsize=(2.2 * n_columns, 2.5 * len(images)), squeeze=False,
    )
    for row_axes, image, true_caption, (words, maps, shares) in zip(axes, images, true_captions, results):
        image = image.permute(1, 2, 0).cpu()
        predicted_caption = ' '.join(words)
        row_axes[0].imshow(image)
        row_axes[0].set_title(
            f'True: {true_caption}\nPredicted: {predicted_caption}', fontsize=7,
            color='green' if predicted_caption == true_caption else 'red',
        )
        for axis, word, attention_map, share in zip(row_axes[1:], words, maps, shares):
            axis.imshow(image)
            # Each cell is drawn around the center of its token's receptive field (the receptive field
            # itself is larger than the cell). Pixels outside of the grid are not covered by any cell.
            axis.imshow(
                attention_map, cmap='inferno', alpha=0.6, interpolation='nearest',
                extent=(start, end, end, start), vmin=0,
            )
            axis.set_xlim(0, image.shape[1])
            axis.set_ylim(image.shape[0], 0)
            axis.set_title(f'"{word}"\non image: {share:.0%}', fontsize=8)
        for axis in row_axes:
            axis.axis('off')
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(figure_name)
    plt.close(figure)


def main(args):
    device = get_device(args)
    image_encoder, decoder, vocabulary, config = load_multi_captioner(args.checkpoint_path, device)
    inverse_vocabulary = {index: word for word, index in vocabulary.items()}
    _, train_objects, held_out_objects = ms.get_objects()
    _, train_eval_dataset, test_dataset = create_datasets(config, train_objects, held_out_objects)
    dataset = test_dataset if args.split == 'held-out' else train_eval_dataset

    indices = random.Random(args.seed).sample(range(len(dataset)), args.n_images)
    images, true_captions = zip(*(dataset[index] for index in indices))
    images = [image.to(device) for image in images]
    with torch.no_grad():
        results = [
            word_attention(image_encoder, decoder, image, vocabulary, inverse_vocabulary, args.layer - 1, args.head)
            for image in images
        ]

    head_name = 'mean over heads' if args.head is None else f'head {args.head}'
    figure_name = args.output or os.path.join(
        os.path.dirname(args.checkpoint_path),
        f'attention_{args.split}_layer{args.layer}' + ('' if args.head is None else f'_head{args.head}') + '.png',
    )
    plot_attention(
        images, true_captions, results, image_encoder.token_grid_extent(config.image_size), figure_name,
        f'Attention to the visual tokens, layer {args.layer}, {head_name}',
    )
    print(f'Saved {figure_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot decoder attention maps of a checkpoint from train_multi_captioner.py'
    )
    parser.add_argument('--checkpoint-path', required=True, help='checkpoint from train_multi_captioner.py')
    parser.add_argument('--layer', type=int, default=1, help='decoder layer, starting at 1')
    parser.add_argument('--head', type=int, help='attention head (starting at 0); default: mean over all heads')
    parser.add_argument('--split', choices=('held-out', 'train'), default='held-out',
                        help='images with a held-out object, or images with training objects only')
    parser.add_argument('--n-images', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0, help='random seed for choosing the images')
    parser.add_argument('--output', help='file name of the figure (default: next to the checkpoint)')
    parser.add_argument('--no-gpu', action='store_true')
    main(parser.parse_args())
