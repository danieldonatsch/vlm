"""Colored shapes dataset generator for VLM experiments."""

import itertools
import math
import random

from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from torchvision import transforms


COLORS = ('red', 'green', 'blue', 'cyan', 'magenta')

SHAPES = ('circle', 'triangle', 'square', 'hexagon', 'star')

POSITIONS = ('top-left', 'top-right', 'bottom-left', 'bottom-right', 'center')

BASE_ALL_COMBINATIONS = tuple(itertools.product(COLORS, SHAPES))

BASE_HELD_OUT_COMBINATIONS = (
    ('red', 'circle'),
    ('green', 'triangle'),
    ('blue', 'square'),
    ('cyan', 'hexagon'),
    ('magenta', 'star'),
)

POSITION_HELD_OUT_COMBINATIONS = tuple(
    combination + (position,)
    for combination, position in zip(
        BASE_HELD_OUT_COMBINATIONS, POSITIONS
    )
)


def get_experiment_combinations(include_position):
    """Return all, training, and held-out combinations for an experiment."""
    if include_position:
        all_combinations = tuple(
            itertools.product(COLORS, SHAPES, POSITIONS)
        )
        held_out_combinations = POSITION_HELD_OUT_COMBINATIONS
    else:
        all_combinations = BASE_ALL_COMBINATIONS
        held_out_combinations = BASE_HELD_OUT_COMBINATIONS
    train_combinations = tuple(
        combination
        for combination in all_combinations
        if combination not in held_out_combinations
    )
    return all_combinations, train_combinations, held_out_combinations


def combination_caption(combination):
    return ' '.join(combination)


def candidate_captions(combinations):
    """Return captions in the same order as the supplied combinations."""
    return tuple(combination_caption(combination) for combination in combinations)


def make_vocabulary(include_position=False) -> dict:
    """Creates a map (dict) which maps each word in the colored shapes dataset to an index.

    :param include_position:
    :return: dict(str: int) Dictionary mapping words to indices.
    """
    words = ('<pad>', '<unk>', *COLORS, *SHAPES)
    if include_position:
        words += POSITIONS
    return {word: index for index, word in enumerate(words)}


def tokenize(caption: str, vocabulary) -> torch.Tensor:
    """Takes a caption, splits it into words and gets for each word the token id.

    :param caption: (str) caption string
    :param vocabulary: (dict) vocabulary mapping words to indices
    :return: (torch.Tensor) tokenized caption
    """
    return torch.tensor(
        [vocabulary.get(word, vocabulary['<unk>']) for word in caption.split()],
        dtype=torch.long,
    )


class ShapeDataset(Dataset):
    """Generate varied images for a fixed set of color-shape combinations."""

    def __init__(self, combinations, samples_per_combination, image_size=64, seed=0):
        self.combinations = combinations
        self.samples_per_combination = samples_per_combination
        self.image_size = image_size
        self.seed = seed
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.combinations) * self.samples_per_combination

    def __getitem__(self, index):
        combination_index = index // self.samples_per_combination
        combination = self.combinations[combination_index]
        sample_seed = self.seed + index
        image = self.make_image(*combination, seed=sample_seed)
        return self.to_tensor(image), combination_caption(combination)

    def make_image(self, color, shape, position=None, seed=0):
        rng = random.Random(seed)
        size = self.image_size
        image = Image.new('RGB', (size, size), 'white')
        draw = ImageDraw.Draw(image)

        object_size = rng.randint(size // 3, size // 2)
        if position is None or position == 'center':
            target_x, target_y = size // 2, size // 2
        else:
            target_x = size * (0.28 if 'left' in position else 0.72)
            target_y = size * (0.28 if 'top' in position else 0.72)
        jitter = size // 16 if position is not None else size // 8
        center_x = int(target_x + rng.randint(-jitter, jitter))
        center_y = int(target_y + rng.randint(-jitter, jitter))
        half_object = object_size // 2
        center_x = max(half_object, min(size - half_object, center_x))
        center_y = max(half_object, min(size - half_object, center_y))
        left = center_x - object_size // 2
        top = center_y - object_size // 2
        right = left + object_size
        bottom = top + object_size

        if shape == 'circle':
            draw.ellipse((left, top, right, bottom), fill=color)
        elif shape == 'square':
            draw.rectangle((left, top, right, bottom), fill=color)
        elif shape == 'triangle':
            draw.polygon(
                ((center_x, top), (left, bottom), (right, bottom)),
                fill=color,
            )
        elif shape in ('hexagon', 'star'):
            outer_radius = object_size / 2
            inner_radius = outer_radius * 0.45
            n_points = 6 if shape == 'hexagon' else 5
            points = []
            for point_index in range(n_points * (1 if shape == 'hexagon' else 2)):
                angle_step = (2 * math.pi / n_points
                              if shape == 'hexagon' else math.pi / n_points)
                angle = -math.pi / 2 + point_index * angle_step
                radius = outer_radius
                if shape == 'star' and point_index % 2:
                    radius = inner_radius
                points.append((
                    center_x + radius * math.cos(angle),
                    center_y + radius * math.sin(angle),
                ))
            draw.polygon(points, fill=color)
        else:
            raise ValueError(f'Unsupported shape: {shape}')
        return image


def collate_shape_batch(batch, vocabulary):
    images, captions = zip(*batch)
    tokenized_captions = [tokenize(caption, vocabulary) for caption in captions]
    token_ids = torch.nn.utils.rnn.pad_sequence(
        tokenized_captions,
        batch_first=True,
        padding_value=vocabulary['<pad>'],
    )
    return torch.stack(images), token_ids
