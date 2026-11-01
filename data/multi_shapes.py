"""Images with two colored shapes side by side, captioned as e.g. "red circle left of green square"."""

import random

from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms

from data.colored_shapes import BASE_ALL_COMBINATIONS, BASE_HELD_OUT_COMBINATIONS, COLORS, SHAPES, draw_shape

# Captions have the form "<color> <shape> left of <color> <shape>"
MAX_CAPTION_WORDS = 6


def get_objects():
    """Returns all, training, and held-out objects (color-shape combinations).

    Held-out objects never appear in training images, not even as one of the two objects.
    """
    held_out_objects = BASE_HELD_OUT_COMBINATIONS
    train_objects = tuple(obj for obj in BASE_ALL_COMBINATIONS if obj not in held_out_objects)
    return BASE_ALL_COMBINATIONS, train_objects, held_out_objects


def make_caption(left_object, right_object) -> str:
    """Canonical caption: the left object is always described first, only "left of" is used."""
    return f'{left_object[0]} {left_object[1]} left of {right_object[0]} {right_object[1]}'


def parse_caption(caption):
    """Inverse of make_caption().

    :return: (tuple) left object and right object as (color, shape) tuples, or None if the caption is malformed.
    """
    words = caption.split()
    if len(words) != MAX_CAPTION_WORDS or words[2:4] != ['left', 'of']:
        return None
    return (words[0], words[1]), (words[4], words[5])


def make_vocabulary() -> dict:
    """Creates a map (dict) which maps each word of the captions (incl. <bos> and <eos>) to an index."""
    words = ('<pad>', '<unk>', *COLORS, *SHAPES, 'left', 'of', '<bos>', '<eos>')
    return {word: index for index, word in enumerate(words)}


class MultiShapeDataset(Dataset):
    """Generates images with two objects, one in the left and one in the right half of the image.

    Every sample is generated from its own seed, so the data set is deterministic.
    """

    def __init__(self, objects, n_samples, image_size=128, seed=0, required_objects=None):
        """
        :param objects: (tuple) (color, shape) combinations the objects are drawn from
        :param n_samples: (int) number of images in the data set
        :param image_size: (int) number of pixels on one side of the square image
        :param seed: (int) random seed
        :param required_objects: (tuple | None) if given, one of the two objects (on a random side)
            is drawn from required_objects instead of objects. Used for the held-out test set.
        """
        self.objects = objects
        self.n_samples = n_samples
        self.image_size = image_size
        self.seed = seed
        self.required_objects = required_objects
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        rng = random.Random(self.seed + index)
        if self.required_objects is None:
            left_object, right_object = rng.choice(self.objects), rng.choice(self.objects)
        else:
            required_object, other_object = rng.choice(self.required_objects), rng.choice(self.objects)
            if rng.random() < 0.5:
                left_object, right_object = required_object, other_object
            else:
                left_object, right_object = other_object, required_object
        image = self.make_image(left_object, right_object, seed=rng.randint(0, 2 ** 31))
        return self.to_tensor(image), make_caption(left_object, right_object)

    def make_image(self, left_object, right_object, seed=0):
        rng = random.Random(seed)
        size = self.image_size
        image = Image.new('RGB', (size, size), 'white')
        draw = ImageDraw.Draw(image)
        jitter = size // 8
        for (color, shape), target_x in ((left_object, size * 0.25), (right_object, size * 0.75)):
            # Same object size in pixels as in the 64 px single-shape data set (size // 3 to size // 2 there),
            # so conv layers pretrained on that data set see objects of the same scale.
            object_size = rng.randint(size // 6, size // 4)
            half_object = object_size // 2
            center_x = int(target_x + rng.randint(-jitter, jitter))
            center_y = int(size / 2 + rng.randint(-jitter, jitter))
            # Keep every object inside its half of the image, so objects never overlap
            # and "left of" is always unambiguous.
            half_start = 0 if target_x < size / 2 else size // 2
            center_x = max(half_start + half_object, min(half_start + size // 2 - half_object, center_x))
            center_y = max(half_object, min(size - half_object, center_y))
            draw_shape(draw, shape, color, center_x, center_y, object_size)
        return image
